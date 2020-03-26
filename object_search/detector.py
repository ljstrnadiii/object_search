import os
import re
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()



class ObjectDetectionModel():
    """Builds an object detection model froma pretrained model.

    Find more models here:
        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

    Attributes:
        frozen_path: str
			path to the frozen model
        class_map_path: str
			a string path to the class map
        graph: tf.graph
            the graph that corresponds to the tf model
        class_map: dict
            the map of int to class names
    """

    def __init__(self, frozen_path, class_map_path):
        """Initialize object, load the model and the class
        map.

        Args:
            frozen_path: str
                the path to the frozen model
            class_map_path: str
                the path to the class map
        """
        self.frozen_path = frozen_path
        self.class_map_path = class_map_path
        # compute graph and class maps
        self.graph = self._load_graph()
        self.class_map = self._get_class_map()

    def _load_graph(self):
        """Loads the graph of a pretrained model"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.frozen_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def _get_class_map(self):
        """Computes the class map"""
        with open(self.class_map_path,'r') as f:
            labs = f.read()

        names = [x.split("display_name")[-1] for x in labs.split("item")[1:]]
        labels = [re.findall(r'"[^"]*"',x)[0][1:-1] for x in names]
        lab_map = dict(enumerate(labels))

        return lab_map

    def __call__(self, image, conf_threshold=.05):
        """Runs inference on an image and filters
        results given a confidence threshold. Lower
        the threshold the more objects pass.

        Args:
            image: numpy matrix of shape (b, w, h, 3)
                batch of images
            conf_threshold: float
                the threshold to drop obejects in (0,1)

        Returns:
            output_dict: dictionary of results
        """
        with self.graph.as_default():
            with tf.compat.v1.Session() as sess:

                # Get handles to input and output tensors
                ops = tf.compat.v1.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                              tensor_name)

                # set input tensor
                image_tensor = tf.compat.v1.get_default_graph()\
                        .get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                     feed_dict={image_tensor: image})

        result = flatten_output_dict(output_dict, self.class_map, conf_threshold)

        return result


def flatten_output_dict(output_dict, class_map, threshold=.05):
    """Takes an output_dict from a typical object detector
    and returns a list of individual output_dicts--one for
    each image in the batch.

    Args:
        output_dict: dict
            The output from an object detector
        threshold: float
            The threshold to filter bounding boxes.
        class_map: dict
            The map from int to string for classes

    Returns:
        of_list: list of dicts
            The 'flattened' output_dictionary
    """
    batch_size = output_dict['num_detections'].shape[0]
    od_list = []

    for i in range(batch_size):
        passing_scores = np.argwhere(output_dict['detection_scores'][i] > threshold).flatten()
        class_ints = output_dict['detection_classes'][i][passing_scores].astype(np.uint8)

        od = {'passing_scores': passing_scores,
              'detection_boxes': output_dict['detection_boxes'][i][passing_scores],
              'detection_scores': output_dict['detection_scores'][i][passing_scores],
              'detection_classes': class_ints,
              'class_names': np.array([class_map[x] for x in class_ints])
        }

        od_list.append(od)

    return od_list
