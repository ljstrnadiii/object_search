from absl import flags
from absl.flags import FLAGS
import os
import uuid

import cv2
import numpy as np

#flags.DEFINE_integer('image_res', 250, 'res of image for object detector.')
#flags.DEFINE_string('output_dir', 'data/discovered_objects/',
#                    'path to store results')

def preprocess(path):
    """Prepreocess data for inference; used with the data generator.
    Args:
        path: str
            path to the image to read

    Returns:
        data: dict
            The dictionary with 'x' as data and 'input' with additional data
    """
    data = {}

    x = cv2.imread(path)
    orig_size = x.shape[:2]
    x = cv2.resize(x, (FLAGS.image_res, FLAGS.image_res))
    x = x.astype(np.float32)

    data['x'] = x
    data['input'] = (path, orig_size)

    return data

def postprocess(batch, labels, results, class_map):
    """Simple function that takes the current batch, labels, and results
    of the object detector and writes out out image in the appropriate
    class directory.

    Args:
        batch: np.ndarray
            The batch of data returned from the dataset iterator
        labels: list
            The list batch of data constructed by making a list
            from the elements of data['inputs'] defined in preprocess
        results: list
            The list of results from the object detector
        class_map: dict
            The int to object name map
    """
    object_slices = []
    object_names = []
    for image, label, res in zip(batch, labels, results):

        # setup image with original aspect ratio
        orig_size = label[1][::-1]
        resized = cv2.resize(image, orig_size).astype(int)

        # get bounding boxes of results and slice of original image
        bboxes = res['detection_boxes']
        mult_hw = np.hstack([orig_size[::-1], orig_size[::-1]])
        new_bboxes = (bboxes * mult_hw).astype(int)

        # get the class names from the class map
        bboxes_classes = res['detection_classes']
        class_names = [class_map[i] for i in bboxes_classes]

        # process each bounding box found
        for bbox, class_name in zip(new_bboxes, class_names):
            xmin, ymin, xmax, ymax = bbox
            sliced = resized[xmin:xmax, ymin:ymax]
            name = class_name

            save_out_slices(sliced, name)

def save_out_slices(sliced, name):
    """Given a list of images, save out under final object_search class
    directory.

    Args:
        object_slices: list
            The list of images that slices of objects found
        object_names: list
            The list of object names that cororesponds to the slices
    """
    object_dir = os.path.join(FLAGS.output_dir, name)
    if not os.path.exists(object_dir):
        os.makedirs(object_dir)
    unique_filename = str(uuid.uuid4())
    image_path = os.path.join(object_dir, unique_filename + '.jpg')
    cv2.imwrite(image_path, sliced)
