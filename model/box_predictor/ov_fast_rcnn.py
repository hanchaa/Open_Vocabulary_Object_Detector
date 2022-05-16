from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers


class OVFastRCNNOutputLayers(FastRCNNOutputLayers):
    def __init__(self,
                 input_shape: ShapeSpec,
                 *,
                 classifier,
                 **kwargs
                 ):
        super().__init__(input_shape, **kwargs)

        del self.cls_score
        self.cls_score = classifier
