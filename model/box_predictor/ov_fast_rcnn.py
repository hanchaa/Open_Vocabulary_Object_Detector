from typing import List, Tuple

import torch
import torch.nn.functional as F
from detectron2.layers import ShapeSpec, cat, cross_entropy
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage

from ..utils import load_class_freq, get_fed_loss_inds


def _log_classification_stats(pred_logits, gt_classes, prefix="fast_rcnn"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/false_negative", num_false_negative / num_fg)


class OVFastRCNNOutputLayers(FastRCNNOutputLayers):
    def __init__(self,
                 input_shape: ShapeSpec,
                 *,
                 classifier,
                 use_binary_ce,
                 cat_freq_path='',
                 fed_loss_freq_weight=0.5,
                 fed_loss_num_cat=50,
                 **kwargs
                 ):
        super().__init__(input_shape, **kwargs)

        del self.cls_score
        self.cls_score = classifier

        self.use_binary_ce = use_binary_ce
        self.fed_loss_num_cat = fed_loss_num_cat

        if use_binary_ce:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.register_buffer("freq_weight", freq_weight)

            if len(self.freq_weight) < self.num_classes:
                self.freq_weight = torch.cat(
                    [self.freq_weight, self.freq_weight.new_zeros(self.num_classes - len(self.freq_weight))]
                )

        else:
            self.freq_weight = None

    def losses(self, predictions, proposals):
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        if self.use_binary_ce:
            loss_cls = self.bce_loss(scores, gt_classes)

        else:
            loss_cls = cross_entropy(scores, gt_classes, reduction="mean")

        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def bce_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        target = torch.zeros_like(pred_class_logits)

        fg_idx = torch.where(gt_classes != self.num_classes)[0]
        target[fg_idx, gt_classes[fg_idx]] = 1

        appeared = get_fed_loss_inds(
            gt_classes,
            num_sample_cats=self.fed_loss_num_cat,
            num_classes=self.num_classes,
            weight=self.freq_weight
        )
        appeared_mask = appeared.new_zeros(self.num_classes + 1)
        appeared_mask[appeared] = 1
        appeared_mask = appeared_mask[:self.num_classes]
        fed_weight = appeared_mask.view(1, self.num_classes).expand(target.shape[0], self.num_classes)

        loss_cls = F.binary_cross_entropy_with_logits(pred_class_logits, target, reduction="none")
        return torch.sum(loss_cls * fed_weight) / target.shape[0]

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)

        if self.use_binary_ce:
            scores = list(scores)
            for idx, s in enumerate(scores):
                s = torch.cat((s, torch.zeros((s.shape[0], 1), dtype=s.dtype, device=s.device)), dim=-1)
                scores[idx] = s

        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_probs(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.
        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_binary_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)

        return probs.split(num_inst_per_image, dim=0)
