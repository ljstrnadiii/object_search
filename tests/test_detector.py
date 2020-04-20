import pytest
import numpy as np
import scipy.misc
import uuid

from object_search.detector import flatten_output_dict

class TestObjectDetector:
    """Tests the object detector."""
    def test_attributes(self, object_detection_model):
        assert hasattr(object_detection_model, 'frozen_path')

    def test_inference(self, object_detection_model, write_images_to_file):
        paths, imgs = write_images_to_file
        # read from paths
        result = object_detection_model(imgs)

        assert type(result) is list
