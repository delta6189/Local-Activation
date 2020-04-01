import torch
import torch.nn as nn
import torchvision
import os

__all__ = ['classifier1', 'classifier2', 'classifier3']

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}

class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
    
class Conv3DWS(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=(3, 3, 3),
                 midplanes=None,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):

        super(Conv3DWS, self).__init__(
            in_planes,
            out_planes,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=(1,2,3,4), keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1,1,1,1,1)+1e-5
        weight = weight / std.expand_as(weight)
        return torch.nn.functional.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)    

    def get_downsample_stride(stride):
        return (stride, stride, stride)
    
class BasicBlock(nn.Module):

    __constants__ = ['downsample']
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, norm='BN'):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        if norm=='BN':
            self.norm1 = nn.BatchNorm3d(planes)
            self.norm2 = nn.BatchNorm3d(planes)
        elif norm=='GN':
            self.norm1 = nn.GroupNorm(16, planes)
            self.norm2 = nn.GroupNorm(16, planes)
                
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            self.norm1,
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            self.norm2,
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out

class WSGNBlock(nn.Module):

    __constants__ = ['downsample']
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(WSGNBlock, self).__init__()
        self.norm1 = nn.GroupNorm(16, planes)
        self.norm2 = nn.GroupNorm(16, planes)

        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes=midplanes, stride=stride),
            self.norm1,
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes=midplanes),
            self.norm2,
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class BasicStem(nn.Sequential):
    """
    The default conv-batchnorm-relu stem
    """
    def __init__(self, standardization=False, norm='BN'):
        if not standardization: 
            conv = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1,3,3), bias=False)
        else: 
            conv = Conv3DWS(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1,3,3), bias=False),
        if norm=='BN': 
            norm = nn.BatchNorm3d(64) 
        elif norm=='GN': 
            norm = nn.GroupNorm(16, 64)
        
        super(BasicStem, self).__init__(
            conv,
            norm,
            nn.ReLU(inplace=True))

class WSGNStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(WSGNStem, self).__init__(
            Conv3DWS(64, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True))        
        
class DilationBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_layer, step=1, WS=False):
        super(DilationBlock, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.convlist = nn.ModuleList()
        for n in range(0, num_layer):
            dilation_t = n * step + 1
            padding_t = dilation_t
            if WS:
                module = nn.Sequential(Conv3DWS(in_ch, 
                                              out_ch, 
                                              kernel_size=(3, 1, 1), 
                                              stride=(1, 1, 1), 
                                              padding=(0, 0, 0), 
                                              dilation=(dilation_t, 1, 1), 
                                              bias=False),
                                      nn.GroupNorm(16, out_ch),
                                      nn.ReLU(inplace=True), 
                                      )
            else:
                 module = nn.Sequential(nn.Conv3d(in_ch, 
                                              out_ch, 
                                              kernel_size=(3, 1, 1), 
                                              stride=(1, 1, 1), 
                                              padding=(0, 0, 0), 
                                              dilation=(dilation_t, 1, 1), 
                                              bias=False),
                                      nn.BatchNorm3d(out_ch),
                                      nn.ReLU(inplace=True), 
                                      )               
            self.convlist.append(module)

    def forward(self, x):
        output_list = []
        
        for layer in self.convlist:
            layer_output = layer(x)
            output_list.append(layer_output)
        
        outputs = torch.cat(output_list, dim=2)

        return outputs
    
class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400,
                 zero_init_residual=False,
                 standardization=False,
                 norm='BN',
                 dilation=False):
        """
        Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1, standardization=standardization, norm=norm)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2, standardization=standardization, norm=norm)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2, standardization=standardization, norm=norm)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2, standardization=standardization, norm=norm)
        
        self.dilation = dilation
        if self.dilation:
            self.dilation1 = DilationBlock(in_ch=64, out_ch=64, num_layer=4, step=4)
            self.dilation2 = DilationBlock(in_ch=64, out_ch=64, num_layer=4, step=4)
            self.dilation3 = DilationBlock(in_ch=64, out_ch=64, num_layer=4, step=4)
            #self.dilation4 = DilationBlock(in_ch=256, out_ch=256, num_layer=4, step=4)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        if self.dilation:
            x = self.dilation1(x)
        x = self.layer2(x)
        if self.dilation:
            x = self.dilation2(x)
        x = self.layer3(x)
        if self.dilation:
            x = self.dilation3(x)
        x = self.layer4(x)
        #if self.dilation:
            #x = self.dilation4(x)
            
        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, standardization=False, norm='BN'):
        ds_stride = conv_builder.get_downsample_stride(stride)
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            if not standardization: 
                conv = nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False)
            else: 
                conv = Conv3DWS(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, padding=0, bias=False)
            if norm=='BN': 
                norm = nn.BatchNorm3d(planes * block.expansion) 
            elif norm=='GN': 
                norm = nn.GroupNorm(16, planes * block.expansion)
            downsample = nn.Sequential(
                conv,
                norm
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)    
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)                
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _video_resnet(arch, pretrained=False, progress=True, standardization=False, norm='BN', **kwargs):
    model = VideoResNet(standardization=standardization, norm=norm, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def r3d_18(pretrained=False, progress=True, **kwargs):
    """
    Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)

def r3d_18ws(pretrained=False, progress=True, **kwargs):
    """
    Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_18ws',
                         pretrained, 
                         progress,
                         block=WSGNBlock,
                         conv_makers=[Conv3DWS] * 4,
                         layers=[2, 2, 2, 2],
                         stem=WSGNStem,
                         norm='GN',
                         **kwargs)

def r3d_18ws_d(pretrained=False, progress=True, **kwargs):
    """
    Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_18ws',
                         pretrained, 
                         progress,
                         block=WSGNBlock,
                         conv_makers=[Conv3DWS] * 4,
                         layers=[2, 2, 2, 2],
                         stem=WSGNStem,
                         norm='GN',
                         dilation=True,
                         **kwargs)

def r3d_98ws(pretrained=False, progress=True, **kwargs):
    """
    Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_98ws',
                         pretrained, 
                         progress,
                         block=WSGNBlock,
                         conv_makers=[Conv3DWS] * 4,
                         layers=[8, 8, 8, 8],
                         stem=WSGNStem,
                         norm='GN',
                         **kwargs)

def r3d_152ws(pretrained=False, progress=True, dilation=False, **kwargs):
    """
    Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_152ws',
                         pretrained, 
                         progress,
                         block=WSGNBlock,
                         conv_makers=[Conv3DWS] * 4,
                         layers=[3, 8, 36, 3],
                         stem=WSGNStem,
                         norm='GN',
                         dilation=dilation,
                         **kwargs)

def r3d_12(pretrained=False, progress=True, **kwargs):
    """
    Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_12',
                         pretrained, 
                         progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[1, 1, 1, 1],
                         stem=BasicStem,
                         **kwargs)

def r3d_12ws(pretrained=False, progress=True, dilation=False, **kwargs):
    """
    Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_12ws',
                         pretrained, 
                         progress,
                         block=WSGNBlock,
                         conv_makers=[Conv3DWS] * 4,
                         layers=[1, 1, 1, 1],
                         stem=WSGNStem, 
                         standardization=True,
                         norm='GN',
                         dilation=dilation,
                         **kwargs)

def r3d_10ws(pretrained=False, progress=True, **kwargs):
    """
    Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_10ws',
                         pretrained, 
                         progress,
                         block=WSGNBlock,
                         conv_makers=[Conv3DWS] * 3,
                         layers=[1, 1, 1],
                         stem=WSGNStem, 
                         standardization=True,
                         norm='GN',
                         **kwargs)

class Classifier1(nn.Module):
    """
    Classifier model # 1
    Attributes:
        Base Architecture: Simple CNN
        # of layer: 
        Type of conv: Usual nn.Conv3D
        Type of norm: None
        Others: -
    """
    def __init__(self):
        super(Classifier1, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))      
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))     
        
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))            
        
        self.conv5 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.fc = nn.Linear(8192, 24)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = inputs # 3*32*112*112
        
        x = self.conv1(x) # 64*32*112*112
        x = self.relu(x) 
        x = self.pool1(x) # 64*32*56*56
        
        x = self.conv2(x) # 128*32*56*56
        x = self.relu(x)
        x = self.pool2(x) # 128*16*28*28
        
        x = self.conv3(x) # 256*16*28*28
        x = self.relu(x)
        x = self.pool3(x) # 256*8*14*14
        
        x = self.conv4(x) # 512*8*14*14
        x = self.relu(x)
        x = self.pool4(x) # 512*4*7*7
        
        x = self.conv5(x) # 512*4*7*7
        x = self.relu(x)
        x = self.pool5(x) # 512*2*4*4 = 16384
        
        x = x.view(-1, 8192)
        x = self.fc(x)
        
        return x
    
class Classifier2(nn.Module):
    """
    Classifier model # 2
    Attributes:
        Base Architecture: Simple CNN
        # of layer: 
        Type of conv: Usual nn.Conv3D
        Type of norm: BatchNorm3D
        Others: -
    """
    def __init__(self):
        super(Classifier2, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))      
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))     
        
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm4 = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))            
        
        self.conv5 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm5 = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.fc = nn.Linear(8192, 24)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = inputs # 3*32*112*112
        
        x = self.conv1(x) # 64*32*112*112
        x = self.norm1(x)
        x = self.relu(x) 
        x = self.pool1(x) # 64*32*56*56
        
        x = self.conv2(x) # 128*32*56*56
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool2(x) # 128*16*28*28
        
        x = self.conv3(x) # 256*16*28*28
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool3(x) # 256*8*14*14
        
        x = self.conv4(x) # 512*8*14*14
        x = self.norm4(x)
        x = self.relu(x)
        x = self.pool4(x) # 512*4*7*7
        
        x = self.conv5(x) # 512*4*7*7
        x = self.norm5(x)
        x = self.relu(x)
        x = self.pool5(x) # 512*2*4*4 = 16384
        
        x = x.view(-1, 8192)
        x = self.fc(x)
        
        return x

class Classifier3(nn.Module):    
    """
    Classifier model # 3
    Attributes:
        Base Architecture: Resnet3D
        # of layer: 12
        Type of conv: Usual nn.Conv3D
        Type of norm: BatchNorm3d
        Others: -
    """
    def __init__(self):
        super(Classifier3, self).__init__()
        self.model = r3d_12()
        self.model.fc = nn.Linear(512, 24)
        
    def forward(self, inputs):
        x = self.model(inputs)

        return x
    
class Classifier4(nn.Module):    
    """
    Classifier model # 4
    Attributes:
        Base Architecture: Resnet3D
        # of layer: 12
        Type of conv: Weight Standardization
        Type of norm: GroupNorm
        Others: -
    """
    def __init__(self):
        super(Classifier4, self).__init__()
        self.model = r3d_12ws()
        self.model.fc = nn.Linear(512, 24)
        
    def forward(self, inputs):
        x = self.model(inputs)

        return x

class Classifier5(nn.Module):
    """
    Classifier model # 5
    Attributes:
        Base Architecture: Resnet3D
        # of layer: 12
        Type of conv: Weight standardization
        Type of norm: GroupNorm
    """    
    def __init__(self):
        super(Classifier5, self).__init__()
        self.head = DilationBlock(in_ch=3, out_ch=64, num_layer=4, step=1)
        self.model = r3d_12ws()
        self.model.fc = nn.Linear(512, 24)
        
    def forward(self, inputs):
        x = self.head(inputs)
        x = self.model(x)
        
        return x       
    
class Classifier6(nn.Module):
    """
    Classifier model # 6
    Attributes:
        Base Architecture: Simple CNN
        # of layer: 
        Type of conv: Usual nn.Conv3D
        Type of norm: BatchNorm3D
        Others: DilationBlock
    """
    def __init__(self):
        super(Classifier6, self).__init__()
        self.dilation = DilationBlock(in_ch=3, out_ch=64, num_layer=3, step=2)
        self.norm1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))      
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))     
        
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm4 = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))            
        
        self.conv5 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm5 = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.fc = nn.Linear(8192, 24)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = inputs # 3*16*112*112
        
        x = self.dilation(x) # 64*48*112*112
        x = self.pool1(x) # 64*48*56*56
        
        x = self.conv2(x) # 128*48*56*56
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool2(x) # 128*24*28*28
        
        x = self.conv3(x) # 256*24*28*28
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool3(x) # 256*12*14*14
        
        x = self.conv4(x) # 512*12*14*14
        x = self.norm4(x)
        x = self.relu(x)
        x = self.pool4(x) # 512*6*7*7
        
        x = self.conv5(x) # 512*6*7*7
        x = self.norm5(x)
        x = self.relu(x)
        x = self.pool5(x) # 512*3*4*4 = 16384

        
        x = x.view(-1, 8192)
        x = self.fc(x)
        
        return x    
    
    
class Classifier7(nn.Module):
    """
    Classifier model # 7
    Attributes:
        Base Architecture: Simple CNN
        # of layer: 
        Type of conv: WS
        Type of norm: GN
        Others: DilationBlock
    """
    def __init__(self):
        super(Classifier7, self).__init__()
        self.dilation = DilationBlock(in_ch=3, out_ch=64, num_layer=3, step=2, WS=True)
        self.norm1 = nn.GroupNorm(16, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = Conv3DWS(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm2 = nn.GroupNorm(16,128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))      
        
        self.conv3 = Conv3DWS(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm3 = nn.GroupNorm(16,256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))     
        
        self.conv4 = Conv3DWS(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm4 = nn.GroupNorm(16,512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))            
        
        self.conv5 = Conv3DWS(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm5 = nn.GroupNorm(16,512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.fc = nn.Linear(8192, 24)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = inputs # 3*16*112*112
        
        x = self.dilation(x) # 64*48*112*112
        x = self.pool1(x) # 64*48*56*56
        
        x = self.conv2(x) # 128*48*56*56
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool2(x) # 128*24*28*28
        
        x = self.conv3(x) # 256*24*28*28
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool3(x) # 256*12*14*14
        
        x = self.conv4(x) # 512*12*14*14
        x = self.norm4(x)
        x = self.relu(x)
        x = self.pool4(x) # 512*6*7*7
        
        x = self.conv5(x) # 512*6*7*7
        x = self.norm5(x)
        x = self.relu(x)
        x = self.pool5(x) # 512*3*4*4 = 16384
        
        x = x.view(-1, 8192)
        x = self.fc(x)
        
        return x    