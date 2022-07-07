import functools
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.yolox import YOLOX
from .sam_detector_mixin import SAMDetectorMixin
from .l2sp_detector_mixin import L2SPDetectorMixin
from mpa.modules.utils.task_adapt import map_class_names
from mpa.utils.logger import get_logger

logger = get_logger()


@DETECTORS.register_module()
class CustomYOLOX(SAMDetectorMixin, L2SPDetectorMixin, YOLOX):
    """SAM optimizer & L2SP regularizer enabled custom YOLOX
    """
    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Hook for class-sensitive weight loading
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                functools.partial(
       