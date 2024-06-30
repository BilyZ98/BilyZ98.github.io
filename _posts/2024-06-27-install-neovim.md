---
layout: post
title: Install neovim with old glibc  
date: 2024-06-25 07:59:00-0400
description: vim 
tags:  vim
categories: vim
featured: false
---




Issue: neovim can not load libc.so.6 which requires
glibc_2.28 which is higher than installed glibc version.



After asking bing chat and find out that 
I need to install latest glibc version with sudo which
I can't I decided to clone neovim and compile locally.

glibc is a system level package.

Previsou release that requires higher glibc version
```
(llmc) [nsccgz_qylin_1@ln101%tianhe2-K nvim-linux64]$ ./bin/nvim --version
./bin/nvim: /lib64/libc.so.6: version `GLIBC_2.28' not found (required by ./bin/nvim)                                                                                   ./bin/nvim: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by ./bin/nvim)                                                                                   (llmc) [nsccgz_qylin_1@ln101%tianhe2-K nvim-linux64]$ ldd ./bin/nvim
./bin/nvim: /lib64/libc.so.6: version `GLIBC_2.28' not found (required by ./bin/nvim)                                                                                   ./bin/nvim: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by ./bin/nvim)                                                                                           linux-vdso.so.1 =>  (0x00007ffc5e491000)
        libm.so.6 => /lib64/libm.so.6 (0x00002ae00f451000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00002ae00f753000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00002ae00f957000)
        libgcc_s.so.1 => /GPUFS/nsccgz_qylin_1/miniconda3/envs/llmc/lib/libgcc_s.so.1 (0x00002ae00eb83000)                                                                      libc.so.6 => /lib64/libc.so.6 (0x00002ae00fb73000)
        /lib64/ld-linux-x86-64.so.2 (0x00002ae00eb4a000)
        libutil.so.1 => /lib64/libutil.so.1 (0x00002ae00ff40000)
```

I installed this release which does not require
glibc.2.31 and now I can run it successfully.
```
https://github.com/neovim/neovim-releases/releases
```

current one 
```
(llmc) [nsccgz_qylin_1@ln101%tianhe2-K nvim-linux64]$ ldd ./bin/nvim
        linux-vdso.so.1 =>  (0x00007ffedf36c000)
        libm.so.6 => /lib64/libm.so.6 (0x00002ba2ec6ee000)                                                                                                                      libdl.so.2 => /lib64/libdl.so.2 (0x00002ba2ec9f0000)
        librt.so.1 => /lib64/librt.so.1 (0x00002ba2ecbf4000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00002ba2ecdfc000)
        libgcc_s.so.1 => /GPUFS/nsccgz_qylin_1/miniconda3/envs/llmc/lib/libgcc_s.so.1 (0x00002ba2ebc2a000)
        libc.so.6 => /lib64/libc.so.6 (0x00002ba2ed018000)                                                                                                                      /lib64/ld-linux-x86-64.so.2 (0x00002ba2ebbf1000)
        libutil.so.1 => /lib64/libutil.so.1 (0x00002ba2ed3e5000)
```

It's too time consuming to build the neovim from scratch.



## .config path for neovim to start with different config
by default, Neovim looks for configuration files in the `~/.config/nvim` directory. However, when you use the `NVIM_APPNAME` environment variable, Neovim will look for the configuration in the `~/.config/{NVIM_APPNAME}` directory instead.

So, if you set `NVIM_APPNAME=lunarvim`, Neovim will look for the configuration files in the `~/.config/lunarvim` directory. Similarly, if `NVIM_APPNAME=nvchad`, it will look in the `~/.config/nvchad` directory.

This allows you to have multiple separate configurations that you can switch between just by changing the `NVIM_APPNAME` environment variable. It's a powerful feature for managing multiple Neovim configurations. 

Remember to clone or place your desired configurations into the respective directories under `~/.config/`. For example, if you're using LunarVim and NvChad, you should have `~/.config/lunarvim` and `~/.config/nvchad` directories, each containing the respective configuration files. 

If you want to use the default configuration, you can just run `nvim` without setting the `NVIM_APPNAME` environment variable, and it will look for the configuration files in the default `~/.config/nvim` directory. 

Please note that the `~/.config` directory is a standard for user-specific application configuration files on Unix-like operating systems. It's defined by the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html). If you want to use a different directory, you'll need to change the `XDG_CONFIG_HOME` environment variable, which defaults to `~/.config`.
