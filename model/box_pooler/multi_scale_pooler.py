from typing import List, Optional

import torch
from detectron2.layers import nonzero_tuple
from detectron2.modeling.poolers import ROIPooler, assign_boxes_to_levels, convert_boxes_to_pooler_format
from detectron2.structures import Boxes


@torch.jit.script_if_tracing
def _create_zeros(
        batch_target: Optional[torch.Tensor],
        channels: int,
        height: int,
        width: int,
        like_tensor: torch.Tensor,
) -> torch.Tensor:
    batches = batch_target.shape[0] if batch_target is not None else 0
    sizes = (batches, channels, height, width)
    return torch.zeros(sizes, dtype=like_tensor.dtype, device=like_tensor.device)


class MultiScaleROIPooler(ROIPooler):
    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
            pooler_type,
            canonical_box_size=224,
            canonical_level=4,
            scale_before_level_assign=False
    ):
        super().__init__(output_size, scales, sampling_ratio, pooler_type, canonical_box_size, canonical_level)

        self.scale_before_level_assign = scale_before_level_assign

    def forward(self, x: List[torch.Tensor], box_lists: List[Boxes]):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.
        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
                len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )
        if len(box_lists) == 0:
            return _create_zeros(None, x[0].shape[1], *self.output_size, x[0])

        scaled_box_lists = []
        for boxes in box_lists:
            scaled_boxes_half_width = (boxes.tensor[:, 2] - boxes.tensor[:, 0]) * (1.5 ** 0.5)
            scaled_boxes_half_height = (boxes.tensor[:, 3] - boxes.tensor[:, 1]) * (1.5 ** 0.5)
            boxes_ctr_x = (boxes.tensor[:, 0] + boxes.tensor[:, 2]) / 2
            boxes_ctr_y = (boxes.tensor[:, 1] + boxes.tensor[:, 3]) / 2

            new_boxes_x1 = torch.clamp_min(boxes_ctr_x - scaled_boxes_half_width, 0)
            new_boxes_y1 = torch.clamp_min(boxes_ctr_y - scaled_boxes_half_height, 0)
            new_boxes_x2 = torch.clamp_max(boxes_ctr_x + scaled_boxes_half_width, x[0].size(3) * (2 ** self.min_level))
            new_boxes_y2 = torch.clamp_max(boxes_ctr_y + scaled_boxes_half_height, x[0].size(2) * (2 ** self.min_level))

            scaled_box_lists.append(Boxes(torch.stack((new_boxes_x1, new_boxes_y1, new_boxes_x2, new_boxes_y2), dim=1)))

        # (M, 5): [batch_index, x0, y0, x1, y1], M = total boxes
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        pooler_fmt_scaled_boxes = convert_boxes_to_pooler_format(scaled_box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        if self.scale_before_level_assign:
            scaled_boxes_level_assignments = assign_boxes_to_levels(
                scaled_box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
            )

        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])
        scaled_output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])

        for level, pooler in enumerate(self.level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            output.index_put_((inds,), pooler(x[level], pooler_fmt_boxes_level))

            # pooling scaled box
            if self.scale_before_level_assign:
                inds = nonzero_tuple(scaled_boxes_level_assignments == level)[0]
            pooler_fmt_scaled_boxes_level = pooler_fmt_scaled_boxes[inds]
            scaled_output.index_put_((inds,), pooler(x[level], pooler_fmt_scaled_boxes_level))

        mean_output = torch.mean(torch.stack((output, scaled_output)), dim=0)

        return mean_output
