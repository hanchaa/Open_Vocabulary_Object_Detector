import clip
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from torch import nn


@BACKBONE_REGISTRY.register()
class CLIPBackbone(Backbone):
    def __init__(self, backbone: str):
        super().__init__()

        if backbone not in ['RN50', 'RN101']:
            raise Exception("backbone should be RN50 or RN101")

        clip_image_encoder = clip.load(backbone)[0].visual

        clip_image_encoder.eval()
        for child in clip_image_encoder.children():
            for param in child.parameters():
                param.requires_grad = False

        self.relu = nn.ReLU()

        self.conv1 = nn.Sequential(
            clip_image_encoder.conv1,
            clip_image_encoder.bn1,
            self.relu,
            clip_image_encoder.conv2,
            clip_image_encoder.bn2,
            self.relu,
            clip_image_encoder.conv3,
            clip_image_encoder.bn3,
            self.relu,
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

        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}

    def output_shape(self):
        return {
            "c2": ShapeSpec(channels=256, stride=4),
            "c3": ShapeSpec(channels=512, stride=8),
            "c4": ShapeSpec(channels=1024, stride=16),
            "c5": ShapeSpec(channels=2048, stride=32)
        }
