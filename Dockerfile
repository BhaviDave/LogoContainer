# Setting base image
FROM python
#pytorch/pytorch:latest

# Copying data to the container
COPY . /flask_api
WORKDIR /flask_api

# Installing dependencies

#RUN pip install opencv-python
#RUN pip install torchvision
RUN pip install opencv-python-headless
RUN pip install flask
RUN pip install requests

RUN pip install Pillow
RUN pip install torchvision
# Detectron2 prerequisites

RUN pip install cython


RUN pip install -U 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install -e detectron2_repo

# Exposing port
EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "run.py" ]


