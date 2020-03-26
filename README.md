# Object Searching in Image Datasets
This is a simple project to find and crop instances of particular objects. 

The basic idea:
dataset of images -> object detector -> crop -> calculate embedding -> store

TODO:
- calculate embeddings with MobileNet V2
- use embeddings to visualize with [tensorflow's projector](https://projector.tensorflow.org/)

## Running the Application with Docker
You can use Docker but building and running the image as follows:

`docker build -t object_search -f Dockerfile .`
`docker run -it object_search`

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
