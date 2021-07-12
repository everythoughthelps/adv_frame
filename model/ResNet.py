import torchvision.models as models

resnet18_CIFAR = models.resnet18(pretrained=False, num_classes=10)
resnet18_ImageNet = models.resnet18(pretrained=True)

