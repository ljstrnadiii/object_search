# Object Searching in Image Datasets
This is a simple project to find and crop instances of particular objects. 

The basic idea:
1. Download an object detector from [Tensorflow's Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
2. Download the [mscoco dataset](http://cocodataset.org/#home)
3. Find all objects in the entire dataset
4. Save to object specific dirs


TODO:
- calculate embeddings with MobileNet V2
- use embeddings to visualize with [tensorflow's projector](https://projector.tensorflow.org/)

## Running the Application with Docker
You can use Docker but building and running the image as follows:

```docker build -t object_search -f Dockerfile .```

```docker run --gpus all -v /home/<usr>/object_search:/home/object_search -it object_search bash run_search.sh```

If you need gpu usage, add the `-gpus all` flag to docker run if you have nvidia-docker installed.

## Directory Structure
- `run_search.sh`: script to run everything from scratch.
	1. download dataset
	2. download model
	3. run python search application
- `object_search/`: 
	- contains modules for models, dataset pipelines, performance, etc. 
- `bin/`:
	- contains some helper scripts
- `data/`: 
	- folder that contains data for project
- `Dockerfile`: (optional)
	- `sudo docker build -t object_search -f Dockerfile .`
- `requirement.txt`: file with required packages

### Also!
Please use the [google python coding style guide](http://google.github.io/styleguide/pyguide.html) for reference. 

Specifically:

1. [Comments and Docstrings](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
2. Follow import style guidelines
3. Focus on readability guidelines for easier collaboration
