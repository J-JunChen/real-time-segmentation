# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ERFNet',
        out_indices=(0, ),
        norm_cfg=norm_cfg,
        ),
    decode_head=dict(
        type='ERFHead',
        in_channels=128,
        mid_channels=64,
        channels=16,
        in_index=-1,
        # dropout_ratio=0.0,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
