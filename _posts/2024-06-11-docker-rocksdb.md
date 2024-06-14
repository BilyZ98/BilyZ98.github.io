---
layout: post
title: Docker RocksDB 
date: 2024-06-11 07:59:00-0400
description: cloud
tags:  cloud docker storage
categories: cloud
featured: false
---

## RocksDB dockerfile
```dockerfile
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
    && rm -rf /var/lib/apt/lists/* \
    && cd /usr/src \
    && git clone --depth 1 --branch ${ROCKSDB_VERSION} https://github.com/facebook/rocksdb.git  && \
    cd rocksdb && \
    make -j4  && \
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

```


## Docker run container
Start with dockerfile
```dockerfile
CMD["./benchmark.sh"]
```

Docker copmose 
```yaml
version: '3.8'
services:
  rocksdb:
    build: .
    container_name: rocksdb
    volumes:
      - ./data:/data
    networks:
      - rocksdb
    command: ["./benchmark.sh", "input.txt"]
    ports:
      - "8080:8080"
    environment:
```

## Issues
db_bench cannot find librocksdb.8.11.so
only librocksdb.a is available and db_bench 
is not compiled statically linked.


Ways to solve this problem
1. link db_bench with librocksdb.a
2. Try cmake

Tried cmake and it compiled successfully.
```bash
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
```


## Specify volume and persist db data
```yaml
version: '3.8'
services:
  rocksdb:
    image: rocksdb
    # build: .
    container_name: rocksdb
    # tty: true
    volumes:
      - db_data:/data
    command: bash -c "cd /usr/src/rocksdb/build && ./db_bench -benchmarks=fillseq -compression-type=none -db=/data/ && tail -f /dev/null"
    ports:
      - "8080:8080"
    working_dir:
      /usr/src/rocksdb/build
    environment:
      - LD_LIBRARY_PATH=/usr/local/lib

volumes:
  db_data:
```

```bash
docker volume list

DRIVER    VOLUME NAME
local     docker_db_data
```

```bash
docker volume inspect docker_db_data

[
    {
        "CreatedAt": "2024-06-14T22:45:30+08:00",
        "Driver": "local",
        "Labels": {
            "com.docker.compose.project": "docker",
            "com.docker.compose.version": "2.20.2",
            "com.docker.compose.volume": "db_data"
        },
        "Mountpoint": "/mnt/nvme1n1/docker-images/volumes/docker_db_data/_data",
        "Name": "docker_db_data",
        "Options": null,
        "Scope": "local"
    }
]
```

