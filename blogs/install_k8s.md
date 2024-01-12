
# Install k8s cluster with 3 ubuntu nodes

## Background 
Recently I encountered a computation issue where there is no enough amount of 
memory to process 100GB of data in memory with pandas.
So I think if I can speed up computation with dask on top of k8s cluster.
So then I follow k8s installation tutorial to set up 3  nodes k8s cluster.
I write down this k8s installation process to help people encoutering GFW issue 
during k8s installtion.


[Follow this doc to install k8s.](https://www.cherryservers.com/blog/install-kubernetes-on-ubuntu)

Things to pay attention to if you are in China and deals with GFW.
1. http_proxy and https_proxy
Please unset your http_proxy and https_proxy environtment vairable 
with following command in bash so that 
kubectl command can works as expected. Becuse kubectl calls curl 
to get information from master nodes and curl will go through 
proxy if if find http_proxy no empty.
```
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
```


2. image_repository in kubeadm
kubeadm needs to pull image from k8s repo which might be blocked 
because of GFW. You can set image_repository option to 
pull image from aliyun in kubeadm init 
command
```
sudo kubeadm init --kubernetes-version v1.28.2 \
--cri-socket=unix:///run/containerd/containerd.sock \
--apiserver-advertise-address=$(hostname -i) \
--control-plane-endpoint=$(hostname -i) \
--apiserver-cert-extra-sans=$(hostname -i) \
--image-repository=registry.aliyuncs.com/google_containers
```

3. containerd image image_repository 
What is containerd ?
This is what bing chat says.
**Containerd** is an industry-standard container runtime with an emphasis on simplicity, robustness, and portability¹². It is available as a daemon for Linux and Windows¹². 

Here are some of its key features:
- Manages the complete container lifecycle of its host system, from image transfer and storage to container execution and supervision to low-level storage to network attachments and beyond¹².
- Supports OCI Image Spec and OCI Runtime Spec¹.
- It is multi-tenant and OCI-compliant¹.

Containerd is designed to be embedded into a larger system, rather than being used directly by developers or end-users². It graduated within the Cloud Native Computing Foundation (CNCF) on February 28, 2019¹.

Source: Conversation with Bing, 1/12/2024
(1) containerd – An industry-standard container runtime with an emphasis on .... https://containerd.io/.
(2) containerd/containerd: An open and reliable container runtime - GitHub. https://github.com/containerd/containerd.
(3) What Is containerd, And How Does It Relate to Docker and Kubernetes?. https://www.howtogeek.com/devops/what-is-containerd-and-how-does-it-relate-to-docker-and-kubernetes/.
(4) Windows container platform | Microsoft Learn. https://learn.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/containerd.

Please follow this doc to update containerd config in before you
start containerd service.

[https://www.cnblogs.com/wod-Y/p/17043985.html](https://www.cnblogs.com/wod-Y/p/17043985.html)











