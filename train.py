# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.
This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.
Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import sys

import wandb
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from hooks.wandb_log import WanDBLog
from data.datasets import lvis_v1

sys.path.insert(0, 'third_party/CenterNet2/')
sys.path.insert(0, 'third_party/Deformable-DETR')
logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)

        if comm.is_main_process():
            wandb.log(ret["bbox"])

        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `common_train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)

    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer=optim,
        first_cycle_steps=cfg.lr_scheduler["first_cycle_steps"],
        cycle_mult=cfg.lr_scheduler["cycle_mult"],
        max_lr=cfg.lr_scheduler["max_lr"],
        min_lr=cfg.lr_scheduler["min_lr"],
        warmup_steps=cfg.lr_scheduler["warmup_steps"],
        gamma=cfg.lr_scheduler["gamma"]
    )

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        optimizer=optim,
        trainer=trainer,
        lr_scheduler=lr_scheduler
    )
    train_hooks = [
        WanDBLog(optimizer=optim) if comm.is_main_process() else None,
        hooks.IterationTimer(),
        hooks.LRScheduler(scheduler=lr_scheduler),
        hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
        if comm.is_main_process()
        else None,
        hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
        hooks.PeriodicWriter(
            default_writers(cfg.train.output_dir, cfg.train.max_iter),
            period=cfg.train.log_period,
        )
        if comm.is_main_process()
        else None,
    ]
    trainer.register_hooks(train_hooks)

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0

    if cfg.wandb.log and comm.is_main_process():
        wandb.init(
            project=cfg.wandb.proj_name,
            group=cfg.wandb.group_name,
            entity=cfg.wandb.entity,
            config=cfg.wandb.config,
            resume=True if args.resume and checkpointer.has_checkpoint() else False,
            id=cfg.wandb.id if args.resume and checkpointer.has_checkpoint() else None
        )

        wandb.watch(model, log="all")

    trainer.train(start_iter, cfg.train.max_iter)

    wandb.finish()


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        do_test(cfg, model)
    else:
        do_train(args, cfg)


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
