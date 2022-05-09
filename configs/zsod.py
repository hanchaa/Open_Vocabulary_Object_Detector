from detectron2.model_zoo import get_config
from detectron2.config import LazyCall

from model.Backbone.CLIPBackbone import CLIPBackbone

default_configs = get_config("new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py")

dataloader = default_configs['dataloader']
model = default_configs['model']
train = default_configs['train']

model.backbone.update(
    bottom_up=LazyCall(CLIPBackbone)(backbone="RN101"),
    in_features=["c2", "c3", "c4", "c5"]
)

