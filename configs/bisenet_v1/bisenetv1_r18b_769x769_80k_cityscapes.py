_base_ = './bisenetv1_r18_769x769_80k_cityscapes.py'
model = dict(pretrained='torchvision://resnet18')
