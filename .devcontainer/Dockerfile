FROM python:3.10

ARG PYTHON_ENV=my_env

ENV PYTHON_ENV=$PYTHON_ENV

# RUN apt-get update && apt-get install libgl1 -y
# RUN apt-get update && apt-get install -y python3-opencv
# RUN pip install opencv-python

RUN mkdir requirements

COPY requirements.txt set_python_env.sh /requirements/

RUN bash ./requirements/set_python_env.sh $PYTHON_ENV
