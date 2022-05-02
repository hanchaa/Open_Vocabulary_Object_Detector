import clip
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()

        clip_image_encoder = clip.load(backbone)[0].visual

        self.conv1 = nn.Sequential(
            clip_image_encoder.conv1,
            clip_image_encoder.bn1,
            clip_image_encoder.relu,
            clip_image_encoder.conv2,
            clip_image_encoder.bn2,
            clip_image_encoder.relu,
            clip_image_encoder.conv3,
            clip_image_encoder.bn3,
            clip_image_encoder.relu,
            clip_image_encoder.avgpool
        )
        self.conv2 = clip_image_encoder.layer1
        self.conv3 = clip_image_encoder.layer2
        self.conv4 = clip_image_encoder.layer3
        self.conv5 = clip_image_encoder.layer4

    def forward(self, x):
        x = x.type(self.conv1[0].weight.dtype)

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        return c2, c3, c4, c5
