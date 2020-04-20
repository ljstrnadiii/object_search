"""Runs the search for objects on a dataset, crops them out, and
stores them in an output directoty."""

from absl import app, flags, logging
from absl.flags import FLAGS
import glob as glob
import os

import cv2
import numpy as np
from tqdm import tqdm

from object_search.detector import ObjectDetectionModel
from object_search.data_generator import BackgroundGenerator
from object_search.data_processors import (
    preprocess,
    postprocess,
    save_out_slices
)

flags.DEFINE_string('dataset', 'data/datasets/mscoco/*/*.jpg',
                    'path to dataset')
flags.DEFINE_integer('subset', None, 'Look at the first subset files only')
flags.DEFINE_string('model_path',
        os.path.join('data/pretrained_models/',
                     'faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb'),
                    'path to an object detector from tf od zoo.')
flags.DEFINE_string('label_map',
                    'data/pretrained_models/mscoco_complete_label_map.pbtxt',
                    'path to the pbtxt label map.')
flags.DEFINE_integer('batch_size', 32, 'batch size for gpu inference.')
flags.DEFINE_integer('n_parallel_pipeline', 8, 'number of processes for '
                     'background data generator.')
flags.DEFINE_integer('n_chunks', 4, 'number of tasks per process for '
                     'background data generator')
flags.DEFINE_integer('image_res', 250, 'res of image for object detector.')
flags.DEFINE_string('output_dir', 'data/discovered_objects/',
                    'path to store results')


def main(_argv):
    # get the dataset images
    files = glob.glob(FLAGS.dataset)
    if FLAGS.subset:
        files = files[:FLAGS.subset]
    logging.info("Dataset size: {}".format(len(files)))

    # load the object detector
    odm = ObjectDetectionModel(FLAGS.model_path, FLAGS.label_map)
    logging.info('Loaded model: {} '
        'with class map here: {}'.format(FLAGS.model_path, FLAGS.label_map))

    # construct background dataset iterator
    dataset = BackgroundGenerator(
        proc=preprocess,
        inputs=files,
        batch_size=FLAGS.batch_size,
        n_parallel=FLAGS.n_parallel_pipeline,
        chunksize=FLAGS.n_chunks
    )
    logging.info('Loaded background data generator '
        '\nbatch size: {} '
        '\nnumber of processes: {} '
        '\nchunksize of {}'.format(FLAGS.batch_size,
                                   FLAGS.n_parallel_pipeline,
                                   FLAGS.n_chunks)
    )

    # operate on the dateset
    pbar = tqdm(total=len(dataset.inputs))
    for batch, labels in dataset:
        # object detection (on gpu)
        results = odm(batch)
        # postprocess (on cpu)
        postprocess(batch, labels, results, odm.class_map)
        pbar.update(len(batch))
    pbar.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass



