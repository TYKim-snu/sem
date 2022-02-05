import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, args=None, backbone='resnet', output_stride=16, num_classes=21,
                 freeze_bn=False, GroupNorm=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if GroupNorm is True:
            GroupNorm = nn.GroupNorm
            BatchNorm = None
            print('using GroupNorm...')
        else:
            #BatchNorm = nn.BatchNorm2d
            BatchNorm = nn.InstanceNorm2d
            GroupNorm = None
        

        print('Norm layer: ', BatchNorm)
        
        self.args = args
        if self.args.backbone == 'resnet_Gen':
            self.backbone = build_backbone(self.args, backbone, output_stride, BatchNorm, GroupNorm=GroupNorm)
        else:
            self.backbone = build_backbone(self.args, backbone, output_stride, BatchNorm, GroupNorm=GroupNorm)
            self.aspp = build_aspp(backbone, output_stride, BatchNorm, GroupNorm=GroupNorm)
            self.decoder = build_decoder(num_classes, backbone, BatchNorm, GroupNorm=GroupNorm)


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input, input_CAD):
        if self.args.backbone == 'resnet_Gen':
            x = self.backbone(input)
            return x
        else:
            x, low_level_feat = self.backbone(input)
            x = self.aspp(x)
            x = self.decoder(x, low_level_feat,input_CAD)
            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                
    def freeze_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
        
    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


