# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class SamsungDataset(CustomDataset):
    """SamsungDataset dataset.
    """
    CLASSES=('Road', 'Sidewalk', 'Construction', 'Fence', 'Pole',
            'Traffic_Light', 'Traffic_sign', 'Nature', 'Sky','Person',
            'Rider', 'Car', 'Background')

    PALETTE=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], 
            [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], 
            [70, 130, 180], [220, 20, 60], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(SamsungDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)