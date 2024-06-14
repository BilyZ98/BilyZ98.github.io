# Use an official Alpine runtime as a parent image
# FROM alpine:3.14
# FROM ubuntu:20.04

# Use an official Ubuntu runtime as a parent image
FROM ubuntu:22.04

# Make sure we don't get notifications we can't answer during building.
# ENV DEBIAN_FRONTEND noninteractive

# Update sources.list
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu\//g' /etc/apt/sources.list

ENV http_proxy http://28.10.10.62:8081
ENV https_proxy http://28.10.10.62:8081 

# RUN echo "deb http://ftp.sjtu.edu.cn/ubuntu focal main universe\n" > /etc/apt/sources.list \
#     && echo "deb http://ftp.sjtu.edu.cn/ubuntu focal-updates main universe\n" >> /etc/apt/sources.list

# Update the system
RUN apt-get update && apt-get upgrade -y

# Set the RocksDB version
ARG ROCKSDB_VERSION=v8.11.3

# Install necessary packages and build RocksDB
RUN apt-get install -y \
    build-essential \
    libgflags-dev \
    libsnappy-dev \
    zlib1g-dev \
    libbz2-dev \
    liblz4-dev \
    libzstd-dev \
    git \
    bash \
    perl \
    cmake \
    && rm -rf /var/lib/apt/lists/* \
    && cd /usr/src \
    && git clone --depth 1 --branch ${ROCKSDB_VERSION} https://github.com/facebook/rocksdb.git  && \
    cd rocksdb && \
    # make -j8 shared_lib && \
    # make -j8 db_bench && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j16 && \
    make install


# RUN apk update && \
#     apk add --no-cache zlib-dev bzip2-dev lz4-dev snappy-dev zstd-dev gflags-dev && \
#     apk add --no-cache build-base linux-headers git bash perl && \
#     mkdir /usr/src && \
#     cd /usr/src && \
#     git clone --depth 1 --branch ${ROCKSDB_VERSION} https://github.com/facebook/rocksdb.git && \
#     cd /usr/src/rocksdb && \
#     sed -i 's/install -C/install -c/g' Makefile && \
#     make -j4 shared_lib && \
#     make install-shared && \
#     apk del build-base linux-headers git bash perl && \
#     rm -rf /usr/src/rocksdb

# Set the working directory
WORKDIR /

