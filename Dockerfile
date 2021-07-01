#FROM python:3-alpine
#FROM bitnami/pytorch

#FROM khammami/detectron2

FROM ylashin/detectron2


#RUN apt-get update
#RUN apt-get install -y git


#RUN pip install Flask-PyMongo
#RUN pip install PyJWT
#RUN pip install flask_uploads
#RUN pip install flask-login

#RUN pip install opencv-python-headless

#RUN apt-get update -y
#RUN apt-get install -y python-pip python-dev build-essential python-opencv

RUN pip install flask
RUN pip install requests

RUN pip install torchvision

RUN pip install Pillow

#RUN pip install opencv-python-headless

#RUN pip install -U 'git+https://github.com/facebookresearch/fvcore'
#RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
#RUN pip install -e detectron2_repo


COPY . /app
WORKDIR /app

# Expose the Flask port
EXPOSE 5000

#CMD [ "python", "-u", "./runaaaa.py" ]

#ENTRYPOINT [ "python" ]
#CMD [ "-u", "run.py", ]

#CMD [ "python", "-u", "run.py" ]

ENTRYPOINT ["python"]
CMD ["run.py"]

#CMD [ "python", "-u", "run.py" ]
