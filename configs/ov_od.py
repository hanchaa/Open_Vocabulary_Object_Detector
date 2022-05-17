import os

import torch
from detectron2.config import LazyCall
from detectron2.data import transforms
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation.lvis_evaluation import LVISEvaluator
from detectron2.layers import ShapeSpec
from detectron2.model_zoo import get_config
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.lr_scheduler import WarmupParamScheduler
from fvcore.common.param_scheduler import CosineParamScheduler

from model.backbone.clip_backbone import CLIPBackbone
from model.box_pooler.multi_scale_pooler import MultiScaleROIPooler
from model.box_predictor.ov_classifier import OpenVocabularyClassifier
from model.box_predictor.ov_fast_rcnn import OVFastRCNNOutputLayers

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
    num_classes=1203,
    box_pooler=LazyCall(MultiScaleROIPooler)(
        output_shape=7,
        scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        sampling_ratio=0,
        pooler_type="ROIAlignV2"
    ),
    box_predictor=LazyCall(OVFastRCNNOutputLayers)(
        input_shape=ShapeSpec(channels=1024, height=None, width=None, stride=None),
        test_score_thresh=0.0001,
        box2box_transform=LazyCall(Box2BoxTransform)(weights=[10, 10, 5, 5]),
        num_classes="${..num_classes}",
        classifier=LazyCall(OpenVocabularyClassifier)(
            input_shape="${..input_shape}",
            num_classes="${..num_classes}",
            prompt_path="prompt/lvis_clip_prompt.npy"
        )
    )
)

image_size = 896

dataloader.train.update(
    dataset={
        "names": "lvis_v1_train_norare"
    },
    sampler=LazyCall(RepeatFactorTrainingSampler)(
        repeat_factors=LazyCall(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
            dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
        )
    ),
    mapper={
        "is_train": True,
        "augmentations": [
            LazyCall(transforms.ResizeScale)(
                min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
            ),
            LazyCall(transforms.FixedSizeCrop)(
                crop_size=[image_size, image_size]
            ),
            LazyCall(transforms.RandomFlip)(
                horizontal=True
            )
        ],
        "image_format": "BGR",
        "use_instance_mask": False,
        "recompute_boxes": True
    },
    total_batch_size=2,
    num_workers=8
)

dataloader.test.dataset.names = "lvis_v1_val"
dataloader.evaluator = LazyCall(LVISEvaluator)(
    dataset_name="${..test.dataset.names}",
    tasks=("bbox",)
)

num_epochs = 200
train.max_iter = (100170 // dataloader["train"]["total_batch_size"]) * num_epochs

lr_multiplier = LazyCall(WarmupParamScheduler)(
    scheduler=CosineParamScheduler(1.0, 0.0),
    warmup_length=500 / train.max_iter,
    warmup_factor=0.067,
)

optimizer = LazyCall(torch.optim.AdamW)(
    params=LazyCall(get_default_optimizer_params)(
        weight_decay_norm=0.0
    ),
    lr=0.0008,
    weight_decay=1e-4,
)

wandb = {'log': True, 'proj_name': 'ov-od', 'group_name': 'ov-od', 'run_name': 'test'}  # add

train.checkpointer.period = 20000
train.output_dir = './output/Lazy/{}'.format(os.path.basename(__file__)[:-3])
