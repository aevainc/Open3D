# FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
FROM ubuntu:18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Regular deps
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    autoconf \
    ca-certificates \
    git \
    gnupg \
    libc++-7-dev \
    libc++abi-7-dev \
    libglu1-mesa-dev \
    libosmesa6-dev \
    libsdl2-dev \
    libtbb-dev \
    libtool \
    libudev-dev \
    libxi-dev \
    ninja-build \
    python3-dev \
    software-properties-common \
    wget \
    xorg-dev \
 && rm -rf /var/lib/apt/lists/*

# CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
 && apt-add-repository --yes 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt-get install -y \
    cmake \
 && rm -rf /var/lib/apt/lists/*

# CCache
RUN git clone https://github.com/ccache/ccache.git \
 && cd ccache \
 && git checkout v4.3 -b 4.3 \
 && mkdir build \
 && cd build \
 && cmake -DCMAKE_BUILD_TYPE=Release -DZSTD_FROM_INTERNET=ON .. \
 && make -j$(nproc) \
 && make install \
 && ccache --version \
 && ccache -M 10G

# Miniconda
# https://towardsdatascience.com/making-docker-and-conda-play-well-together-eda0ff995e3c
# https://stackoverflow.com/a/58269633/1255535
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && which conda \
    && conda --version

RUN conda create -y -n open3d3 python=3.9
SHELL ["conda","run","-n","open3d3","/bin/bash","-c"]
RUN which python \
 && python --version \
 && which pip \
 && pip --version

# Install ML deps
# No virtual environment is needed, default to python 3.6
RUN python -m pip install -U pytest scipy
RUN python -m pip install -U tensorflow==2.5.0
RUN python -m pip install -U https://github.com/intel-isl/open3d_downloads/releases/download/torch1.7.1/torch-1.7.1-cp39-cp39-linux_x86_64.whl

# Ignore docker cache from now on
# https://stackoverflow.com/a/58801213/1255535
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

# Copy Open3D
COPY . /root/Open3D
WORKDIR /root/Open3D
