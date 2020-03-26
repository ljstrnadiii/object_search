# Ubuntu 18.04 included below
FROM tensorflow/tensorflow:1.14.0-gpu-py3

# Install object detection api dependencies
RUN apt-get update
RUN apt-get install -y protobuf-compiler python-lxml git && \
    pip install Cython && \
    pip install contextlib2 && \
    pip install pycocotools

RUN git clone --depth 1 https://github.com/tensorflow/models.git
RUN cp -r /models /home

# Run protoc on the object detection repo
RUN cd /home/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

# Set the PYTHONPATH to finish installing the API
ENV PYTHONPATH $PYTHONPATH:/home/models/research:/home/models/research/slim

RUN apt-get update
RUN apt-get -y install cmake 
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# expose ports for jupyter notebooks and tensorboard
EXPOSE 8888
EXPOSE 6006

RUN mkdir /home/object_search
WORKDIR /home/object_search

# copy the current dir
COPY . .

# install the requirements.txt packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# install the package
RUN pip install -e .

# add some jupyter configurations
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py


CMD ["/bin/bash"]
