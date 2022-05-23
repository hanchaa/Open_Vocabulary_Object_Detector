from detectron2.engine import HookBase
from detectron2.utils.events import get_event_storage
import wandb


class WanDBLog(HookBase):
    def __init__(self, optimizer):
        self.__optimizer = optimizer

    def after_step(self):
        storage = get_event_storage()
        log = storage.latest()
        wandb.log({
            "total_loss": log["total_loss"][0],
            "loss_cls": log["loss_cls"][0],
            "loss_box_reg": log["loss_box_reg"][0],
            "loss_rpn_cls": log["loss_rpn_cls"][0],
            "loss_rpn_loc": log["loss_rpn_loc"][0],
            "lr": self.__optimizer.param_groups[0]["lr"]
        })
