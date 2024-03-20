import torch
import torch.nn as nn
import kornia.augmentation as ka
from torchvision.models import resnet18


class SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLR, self).__init__()
        self.normalizer = Normalizer()
        self.augmentor = Augmentor()

        self.backbone = resnet18(weights='DEFAULT')
        input_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim),)

    def _double_aug(self, xs):
        # repeat xs along the batch dimension
        xss = xs.repeat(2, 1, 1, 1)
        xss = self.augmentor(xss)
        return xss

    def forward(self, xs):
        xss = self._double_aug(xs)

        xss = self.backbone(xss)
        xss = self.projector(xss)
        return xss

    @torch.no_grad()
    def feature(self, xs):
        xs = self.normalizer(xs)
        xs = self.backbone(xs)
        return nn.functional.normalize(xs, dim=-1)


class Augmentor(nn.Module):
    def __init__(self):
        super(Augmentor, self).__init__()
        self.aug = nn.Sequential(
            ka.Resize(size=(256, 256)),
            ka.CenterCrop(size=(224, 224)),
            ka.RandomResizedCrop(
                (224, 224), scale=(0.2, 1.),
                resample="nearest",
                cropping_mode="resample"),
            ka.RandomHorizontalFlip(),
            ka.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            ka.RandomGrayscale(p=0.2),
            ka.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        )

    @torch.no_grad()
    def forward(self, x):
        return self.aug(x)


class Normalizer(nn.Module):
    def __init__(self):
        super(Normalizer, self).__init__()
        self.transform = nn.Sequential(
            ka.Resize(size=(256, 256)),
            ka.CenterCrop(size=(224, 224)),
            ka.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        )

    @torch.no_grad()
    def forward(self, x):
        return self.transform(x)
