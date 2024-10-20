---
layout: post
title:  ssh display image on local server 
date: 2024-10-17 07:59:00-0400
description:  
tags:   ssh  
categories: ssh
featured: false
---



To display an image on your local machine after using the `ssh -Y` command, you'll need to have an image viewer installed on the remote server. Here's a step-by-step example:

1. **Connect to the remote server with X11 forwarding**:
   ```sh
   ssh -Y user@remote-server
   ```

2. **Check if an image viewer is installed**:
   - Common image viewers include `eog` (Eye of GNOME), `display` (ImageMagick), and `xdg-open`.
   - You can check if one of these is installed by running, for example:
     ```sh
     which eog
     ```
     If it's installed, the command will return the path to the executable. If not, you might need to install it using the package manager for your distribution.

3. **Display an image**:
   - Assuming you have an image file named `example.png` in your home directory on the remote server, you can display it using:
     ```sh
     eog ~/example.png
     ```
   - This command will open the image using the Eye of GNOME image viewer, and the image should be displayed on your local machine.

Remember, the exact command to display an image will depend on the image viewers available on the remote server. If you're not sure which image viewers are installed, you can try running the command for a few common ones until you find one that works. Also, ensure that your local machine is running an X server (like XQuartz on macOS or Xming on Windows) to handle the forwarded X11 display.
