# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='LINKHead',
        in_channels=(64, 128, 256, 512),
        in_index=(0, 1, 2, 3),
        input_transform='multiple_select',
        with_classifier=True, # if True, the channels will be 32, otherwise, channels will be 64.
        channels=32,
        # channels=64,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        # upsample_cfg=dict(type='InterpConv', conv_first=True),
        upsample_cfg=dict(type='DeconvModule'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
