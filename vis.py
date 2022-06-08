from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

import torch
import cv2
import numpy as np

import pathlib
import json


def main(args):
    with open("datasets/lvis/lvis_v1_val.json") as f:
        lvis = json.load(f)
    categories = [{"name": i["name"], "freq": i["frequency"]} for i in lvis["categories"]]

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval()
    test_loader = instantiate(cfg.dataloader.test)

    iterator = iter(test_loader)
    for _ in range(5):
        inputs = next(iterator)

        path = f"fig/{inputs[0]['file_name'].split('/')[-1].split('.')[0]}"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        img = cv2.imread(inputs[0]["file_name"])
        for annotation in lvis["annotations"]:
            if annotation["image_id"] != inputs[0]["image_id"]:
                continue

            box = annotation["bbox"]
            color = np.random.randint(256, size=(3,)).tolist()
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[0] + box[2]), int(box[1] + box[3]))

            cv2.rectangle(img, pt1, pt2, color, 2)

            size, _ = cv2.getTextSize(categories[annotation["category_id"] - 1]["name"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, pt1, (pt1[0] + size[0], pt1[1] + size[1]), color, thickness=-1)
            cv2.putText(img, categories[annotation["category_id"] - 1]["name"], (pt1[0], pt1[1] + size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(f"{path}/gt.png", img)
        with torch.no_grad():
            outputs = model(inputs)


        img = cv2.imread(inputs[0]["file_name"])
        instances = outputs[0]["instances"].get_fields()

        for idx, (box, score, pred_class) in enumerate(zip(instances["pred_boxes"], instances["scores"], instances["pred_classes"])):
            temp = cv2.imread(inputs[0]["file_name"])

            color = (102, 153, 255) if categories[pred_class]["freq"] == "r" else (204, 255, 51)
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))

            cv2.rectangle(img, pt1, pt2, color, 2)
            cv2.rectangle(temp, pt1, pt2, color, 2)
            cv2.imwrite(f"{path}/{idx:2d}.png", temp)

            with open(f"{path}/predicted.txt", "a") as f:
                f.write(f"{idx} {categories[pred_class]['name']} {score * 100:.2f}%\n")

        cv2.imwrite(f"{path}/all.png", img)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )