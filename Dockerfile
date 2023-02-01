FROM nvcr.io/nvidia/pytorch:22.12-py3


ENV DEBIAN_FRONTEND=noninteractive

ENV MPLBACKEND=agg

RUN apt-get update && \
        apt-get install -y \
        git \
        wget \
        unzip \
        vim \
        zip \
        curl \
        yasm \
        pkg-config \
        nano \
        tzdata \
        ffmpeg \
        libgtk2.0-dev \
        libgl1-mesa-glx && \
    rm -rf /var/cache/apk/*

RUN pip install --upgrade pip

RUN pip --no-cache-dir install \
      Cython==0.29.21

RUN pip --no-cache-dir install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


RUN pip --no-cache-dir install \
      numpy \
	matplotlib \
	tqdm \
	imageio \
        pillow \
	opencv-contrib-python \
	tensorboard \
	pyyaml \
      neptune-client

#RUN pip install timm

RUN git clone https://github.com/rwightman/pytorch-image-models.git && cd pytorch-image-models && pip install -e .
RUN ln -sf /usr/share/zoneinfo/Turkey /etc/localtime

WORKDIR /workspace
