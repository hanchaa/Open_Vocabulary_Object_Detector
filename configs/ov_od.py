from detectron2.model_zoo import get_config
from detectron2.config import LazyCall

from model.backbone.clip_backbone import CLIPBackbone
from model.box_pooler.multi_scale_pooler import MultiScaleROIPooler

default_configs = get_config("new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py")

dataloader = default_configs['dataloader']
model = default_configs['model']
train = default_configs['train']

model.backbone.update(
    bottom_up=LazyCall(CLIPBackbone)(
        backbone="RN101", norm="SyncBN"
    ),
    in_features=["c2", "c3", "c4", "c5"]
)

[model.roi_heads.pop(k) for k in ["mask_in_features", "mask_pooler", "mask_head"]]

model.roi_heads.update(
    box_pooler=LazyCall(MultiScaleROIPooler)(
        output_shape=7,
        scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        sampling_ratio=0,
        pooler_type="ROIAlignV2"
    )
)

print(model)
