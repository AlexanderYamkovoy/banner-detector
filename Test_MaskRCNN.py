import unittest
from MaskRCNNInsertor import MRCNNLogoInsertion
import yaml


class TestMaskRCNN(unittest.TestCase):

    def setUp(self):
        self.mask_rcnn_insertor = MRCNNLogoInsertion()
        self.mask_rcnn_insertor.init_params('template.yaml')

    def test_init_params(self):
        # self.assertIsNone(self.mask_rcnn_insertor.init_params('template.yaml'))
        self.assertIsNotNone(self.mask_rcnn_insertor.config)

    def test_detect_banner(self):
        self.mask_rcnn_insertor.detect_banner()
        self.assertIsNone(self.mask_rcnn_insertor.detect_banner())


if __name__ == '__main__':
    unittest.main()
