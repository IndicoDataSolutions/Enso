FROM ubuntu:16.04

RUN mkdir /enso
ADD ./requirements.txt /enso/requirements.txt
RUN apt-get update
RUN apt-get install -y python3-pip libtiff5-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python-tk
RUN pip3 install -r /enso/requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ADD . /enso
WORKDIR /enso

RUN python3 setup.py develop

CMD ["/bin/bash"]
