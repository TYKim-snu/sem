from modeling.backbone import resnet, xception, drn, mobilenet, resnet_Gen, resnet_nonlocal, resnet_CBAM

def build_backbone(args, backbone, output_stride, BatchNorm, GroupNorm=None):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained=False)
    elif backbone == 'resnet50':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained=False)
    elif backbone == 'resnet50_nonlocal':
        return resnet_nonlocal.ResNet50(output_stride, BatchNorm, pretrained=False)
    elif backbone == 'resnet50_CBAM':
        return resnet_CBAM.ResNet50(output_stride, BatchNorm, pretrained=True, pretrained_path=args.backbone_pretrained)
    elif backbone == 'resnet101_CBAM':
        return resnet_CBAM.ResNet101(output_stride, BatchNorm, pretrained=True,
                                     pretrained_path=args.backbone_pretrained)
    elif backbone == 'resnet_Gen':
        if GroupNorm is None:
            return resnet_Gen.ResnetGenerator(input_nc=3, output_nc=1, norm_layer=BatchNorm)
        elif GroupNorm is not None:
            return resnet_Gen.ResnetGenerator(input_nc=3, output_nc=1, norm_layer=GroupNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, GroupNorm=GroupNorm, pretrained=False)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, GroupNorm=GroupNorm, pretrained=False)
    else:
        raise NotImplementedError
