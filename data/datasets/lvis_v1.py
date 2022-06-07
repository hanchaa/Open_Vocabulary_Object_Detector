import os

from detectron2.data.datasets import register_lvis_instances
from detectron2.data.datasets.lvis import get_lvis_instances_meta

register_lvis_instances(
    "lvis_v1_train_norare",
    get_lvis_instances_meta("lvis_v1_train_norare"),
    os.path.join("datasets", "lvis/lvis_v1_train_norare.json"),
    os.path.join("datasets", "coco/")
)
