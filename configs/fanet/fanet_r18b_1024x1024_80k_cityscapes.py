_base_ = './fanet_r18_1024x1024_80k_cityscapes.py'
model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet'),
    )
