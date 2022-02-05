import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    def __init__(self, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.SpatialGate = SpatialGate()
    
    def forward(self, x):
        x_out = self.SpatialGate(x)
        return x_out

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv1= nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, 
                        dilation=dilation, groups=groups, bias=bias)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, 
                        dilation=dilation, groups=groups, bias=bias)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, 
                        dilation=dilation, groups=groups, bias=bias)
        
        self.compress = ChannelPool()
        
        self.last_conv = nn.Conv2d(2, out_planes, kernel_size=1, stride=stride, padding=0, 
                        dilation=dilation, groups=groups, bias=bias)
        
        #self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.bn = nn.InstanceNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_final = torch.cat((x1,x2,x3),dim=1)
        x_final = self.compress(x_final)
        x_final = self.last_conv(x_final)

        if self.bn is not None:
            x_final = self.bn(x_final)
        if self.relu is not None:
            x_final = self.relu(x_final)
        return x_final

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )