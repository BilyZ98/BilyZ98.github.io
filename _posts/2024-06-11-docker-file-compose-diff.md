---
layout: post
title: Docker beginner
date: 2024-06-11 07:59:00-0400
description: cloud
tags:  cloud docker   
categories: cloud
featured: false
---


## Difference between Dockerfile and Docker Compose 
Dockerfile and Docker Compose are both important components of Docker, but they serve different purposes:

1. **Dockerfile**¹²³⁴⁵:
   - A Dockerfile is a text file that contains instructions for building a Docker image¹²³⁴⁵.
   - It's essentially Docker's version of a Makefile or build.gradle⁵.
   - It includes commands like `RUN`, `ADD`, `COPY`, `ENTRYPOINT`, etc².
   - You build an image from a Dockerfile².
   - An image is a template for running containers, consisting of multiple read-only layers².
   - Dockerfile is used to create a container image².

2. **Docker Compose**¹²³⁴⁵:
   - Docker Compose is a tool for defining and running multi-container Docker applications¹²³⁴⁵.
   - It uses a YAML file (docker-compose.yml) to define the services that make up your app so they can be run together in an isolated environment³.
   - A single Compose file can contain multiple container deployment configurations².
   - Docker Compose is used to deploy an instance of that image as a container².
   - It's more readable, more maintainable, more intuitive².

In summary, a Dockerfile is used to build images, while Docker Compose is used to define and run multi-container applications¹²³⁴⁵. The Dockerfile uses the `docker build` command, while the docker-compose.yml file uses the `docker-compose up` command⁴. A docker-compose.yml file can reference a Dockerfile, but a Dockerfile can’t reference a docker-compose file⁴.


## Specify docker image storage location
To specify a different storage location for Docker images, you need to modify Docker's configuration. Here are the steps for Linux and Windows:

**For Linux**¹²:
1. Stop the Docker daemon if it's running. You can do this with the command: `sudo systemctl stop docker`.
2. Create a `daemon.json` file in `/etc/docker/` if it doesn't exist¹.
3. Open the `daemon.json` file and add the following, replacing `"/mnt"` with your desired path¹:

```json
{
  "data-root": "/mnt"
}
```

4. Save and close the file.
5. Start the Docker daemon with the command: `sudo systemctl start docker`.

**For Windows**³:
1. Quit Docker Desktop if it's running.
2. Open the `daemon.json` file located at `C:\ProgramData\Docker\config\`.
3. Add the following to the file, replacing `"D:\\Virtual Machines\\Docker"` with your desired path³:

```json
{
  "data-root": "D:\\Virtual Machines\\Docker"
}
```

4. Save and close the file.
5. Start Docker Desktop.

After making these changes, Docker will store its images in the specified directory. You can confirm the new storage location by running the command `docker info` and checking the `Docker Root Dir` value¹²³. Please note that you need to have the necessary permissions to read and write to the specified directory¹²³.


## Docker remove container and image
```bash
# show all containers
docker ps -a

docker rm <container_id>
docker rmi <image_id>
```

## Docker pull image with proxy
1. Create a systemd drop-in directory for the Docker service if it doesn't exist²⁴:

```bash
mkdir /etc/systemd/system/docker.service.d
```

2. Create a file called `/etc/systemd/system/docker.service.d/http-proxy.conf`²⁴. Add the following content to the file, replacing `proxy.example.com:80` with your proxy host and port²⁴:

```bash
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:80/"
Environment="HTTPS_PROXY=http://proxy.example.com:80/"
```

If you have internal Docker registries that you need to contact without proxying, you can specify them via the `NO_PROXY` environment variable²:

```bash
Environment="NO_PROXY=localhost,127.0.0.0/8,docker-registry.somecorporation.com"
```

3. Reload the systemd daemon to apply the changes²⁴:

```bash
sudo systemctl daemon-reload
```

4. Restart the Docker service²⁴:

```bash
sudo systemctl restart docker
```

Now, Docker will use the specified proxy when pulling images²⁴.

Remember, you need to have the necessary permissions to create and modify files in `/etc/systemd/system/docker.service.d`²⁴. If you don't, you may need to use `sudo` or log in as root²⁴. Also, ensure that your proxy server is properly configured and reachable from your Docker host²⁴.

## Ubuntu repository mirror in China
http://ftp.sjtu.edu.cn/ubuntu/

## Solve unable to connect to archive.ubuntu.com during docker build 
Solution I tried but not work:
https://gist.github.com/dyndna/12b2317b5fbade37e747

https://stackoverflow.com/questions/24991136/docker-build-could-not-resolve-archive-ubuntu-com-apt-get-fails-to-install-a

Tried replacing sources.list with tsinghua source list but not work.

Tried to pull the image and enter the conter to see if I can ping
Failed.
```
docker run -it ubuntu bash
```

Tried solution in this doc
https://talk.plesk.com/threads/docker-is-unable-to-connect-to-the-internet.370357/

Tried ping and curl on host machine
ping does not work but curl can work.
So I think I found the root cause.
I set proxy for curl but I did not set proxy for ping.

And I should set proxy for container so 
that it can do `apt-get update` successfully.
Add these lines to set proxy that is used in host machine for container.
```dockerfile
# Update sources.list
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu\//g' /etc/apt/sources.list

ENV http_proxy http://28.10.10.62:8081
ENV https_proxy http://28.10.10.62:8081 

```

## docker remove all containers
```bash
# show all containers
docker ps -a

docker rm $(docker ps -a -q -f status=exited)
```
