# Ubuntu 18.04 included below
FROM tensorflow/tensorflow:1.14.0-gpu-py3

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
