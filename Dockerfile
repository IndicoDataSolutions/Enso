FROM tensorflow/tensorflow:latest-gpu-py3

RUN mkdir /enso
ADD ./requirements.txt /enso/requirements.txt
RUN pip3 install -r /enso/requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ADD . /enso
WORKDIR /enso
RUN python3 setup.py develop

CMD ["/bin/bash"]
