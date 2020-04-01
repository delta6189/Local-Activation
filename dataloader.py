import torch
import torch.nn as nn
import cv2
import os, sys
import numpy as np
import opencv_transforms.transforms as TF
import opencv_transforms.functional as FF
import random
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from mypath import Path

__all__ = [
    'VideoFolder'
]

def has_file_allowed_extension(filename, extensions):
    """
    Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    data = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    data.append(item)

    return data

def make_tensor(img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = img.astype(np.float64)
    #img -= np.array([[[90.0, 98.0, 102.0]]])
    img = torch.from_numpy(img)
    img = img.permute((3, 0, 1, 2)).to(device)
    return img

def video_loader(path, transform, length, sampling_rate, start_random):
    cap = cv2.VideoCapture(path)
    frames = []
    cap_length = 0
    iters = 0
    # Measure entire video length
    cap_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_random:
        # Set start & end point randomly
        start = random.randint(0, cap_length - length * sampling_rate)
        end = start + length * sampling_rate
        
    else:
        start = 0
        end = start + length * sampling_rate
    
    # Cut video and convert to numpy array
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret or iters == end:
            break
        if iters >= start and iters % sampling_rate==0: 
            if transform is not None:
                frame = transform(frame)
            frames.append(frame)
        iters += 1

    cap.release()
    while len(frames)<length:
        frames.append(frames[-1])
    video = np.stack(frames)
    video = crop(video, crop_size = 112)
    video = make_tensor(video)

    return video

def to_onehot(label, num_class=24):
    onehot = torch.zeros(num_class)
    onehot[label] = 1
    
    return onehot.long()

def crop(video, crop_size):
    height_index = np.random.randint(video.shape[1] - crop_size)
    width_index = np.random.randint(video.shape[2] - crop_size)
    video = video[:,
                  height_index:height_index + crop_size,
                  width_index:width_index + crop_size, 
                  :]
    return video

class VideoFolder(VisionDataset):
    """
    A generic data loader where the samples are arranged in this way: ::
    
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
    
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions. 
            * Note : Both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file and check if the file is a valid file (used to check of corrupt files)
            * Note : Both extensions and is_valid_file should not be passed.
        clip_length(int): Length of clips
            
     Attributes:
     
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
        
    """

    def __init__(self, 
                 root, 
                 loader=video_loader, 
                 extensions=('.avi'), 
                 transform=None, 
                 target_transform=None, 
                 is_valid_file=None,
                 clip_length=16,
                 sampling_rate=4,
                 start_random=False):
        super(VideoFolder, self).__init__(root, transform=transform,target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.clip_length = clip_length
        self.sampling_rate = sampling_rate
        self.start_random = start_random

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        
        Args:
            dir (string): Root directory path.
            
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
            
        Ensures:
            No class is a subdirectory of another.
            
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
            
        """
        path, target = self.samples[index]
        sample = self.loader(path, 
                             self.transform, 
                             length=self.clip_length, 
                             sampling_rate=self.sampling_rate, 
                             start_random=self.start_random)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
    
def make_clips(video, length=64):
    """
    Cut input video into clips time-uniformly.
    
    Args:
        video (tensor): a 4D tensor which has form of [C, L, H, W]
        length (integer): # of clip frame

    Returns:
        clip_list: a list of clips 
    """
    clip_list = torch.split(video, length, dim=1)
    clip_list = list(clip_list)
    
    return clip_list

def play_video(video):
    video = np.asarray(video.permute(1, 2, 3, 0).detach().cpu())
    duration = video.shape[0]

    for t in range(0, duration):
        frame = video[t] / 255.0
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(33) & 0xFF == 27: # Press ESC to close window
            break

    cv2.destroyAllWindows()

class VideoDataset(Dataset):
    def __init__(self, dataset='aps', split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in sorted(os.listdir(os.path.join(folder, label))):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if not os.path.exists('dataloaders/aps_labels.txt'):
            with open('dataloaders/aps_labels.txt', 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id+1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
   
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_preprocess(self):
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False
        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '%05d.jpg'%(i+1)), img=frame)
                i += 1
            count += 1

        capture.release()

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            #frame -= np.array([[[90.0, 98.0, 102.0]]])
            frame
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float64'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, 
                 :]

        return buffer
