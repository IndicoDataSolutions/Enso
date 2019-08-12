FROM tensorflow/tensorflow:1.13.1-gpu-py3


# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=10.0" \
    LANG=C.UTF-8 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NLTK_DATA=/usr/local/nltk-data


RUN mkdir /enso
ADD ./requirements.txt /enso/requirements.txt
RUN apt-get update
RUN apt-get install -y python3-pip libtiff5-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python-tk
RUN apt-get install html5lib
RUN pip3 install --upgrade pip
RUN pip3 install -r /enso/requirements.txt
RUN python3 -m spacy download en

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ADD . /enso
WORKDIR /enso

RUN python3 setup.py develop

CMD ["sleep","infinity"]
