import os
import uuid
import pytest
import numpy as np
import cv2

from object_search.detector import ObjectDetectionModel

@pytest.fixture(scope="module")
def object_detection_model():
    model = ObjectDetectionModel(
        frozen_path=os.path.join(
            '../data/pretrained_models/',
            'faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb'
        ),
        class_map_path='../data/pretrained_models/mscoco_complete_label_map.pbtxt',
    )
    return model

@pytest.fixture(scope="session")
def write_images_to_file(tmpdir_factory):
    n_images=3
    x = np.zeros((160,160,3))
    paths = []
    # fake image tp tempdir multiple times
    for i in range(n_images):
        unique_path = str(uuid.uuid4())
        path = str(tmpdir_factory.mktemp("data")\
                .join("{}.png".format(unique_path)))
        cv2.imwrite(path, x)
        paths.append(path)

    return paths, np.zeros((n_images,160,160,3))

