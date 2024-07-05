---
# Common-Defined params
title: "Nvidia Jetson and Ollama"
date: "2024-07-02"
lastmod: "2024-07-04"
description: "LLMs in a single board computer, with GPU"
lead: "How to get up and running an Ollama over a Nvidia Xavier NX" # Lead text
thumbnail: "img/posts/davis-vargas-2vSNlKHn9h0-unsplash.jpg" # Thumbnail image
lang: en
categories:
  - "AI"
tags:
  - "Xavier"
  - "Nvidia"
  - "VSCodium"
draft: false
asciinema: true
menu: side # Optional, add page to a menu. Options: main, side, footer
comments: false # Enable Disqus comments for specific page
authorbox: true # Enable authorbox for specific page
pager: true # Enable pager navigation (prev/next) for specific page
toc: true # Enable Table of Contents for specific page
sidebar: "right" # Enable sidebar (on the right side) per page
widgets: # Enable sidebar widgets in given order per page
  - "search"
  - "recent"
  - "taglist"
  - "social"
---

Photo by [DAVIS VARGAS](https://unsplash.com/@davacorp?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash) on [Unsplash](https://unsplash.com/photos/brown-llama-on-green-grass-field-during-daytime-2vSNlKHn9h0)
  

Running a local LLM in a Single-Board-Computer (SBC) for 10W is fun and cheap, the results? good enough (for fun, no profit). This is a personal reminder of how to get it done.

<!--more-->

## Intro - Skip this

After my [Masters'](/about) degree I wanted to try some ML models, specially those that would allow me to [write](https://www.activestate.com/blog/how-to-monitor-social-distancing-using-python-and-object-detection/) about them, given that my laptop wasn't powerful enough I bought a [Xavier developer kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-series/) with some of the money that I got from tech writing. 

Years later, I re-discovered the SBC getting dust in my apartment and decided to give it a new life as my personal LLM server. Ollama, docker and open source models allows me to have fun for a weekend.

## Flashing the Nvidia Xavier

SBC works out-of-the-box with an SD card that is used as primary storage and boot device, this is an easy way to start but, it's terribly slow and storage space is limited. Instead you can [install an NVME disk](https://medium.com/@ramin.nabati/installing-an-nvme-ssd-drive-on-nvidia-jetson-xavier-37183c948978) directly into the board to be used as primary boot device since Linux For Tegra (L4T) version 4.4. 

To flash the [latest](https://developer.nvidia.com/embedded/jetson-linux-archive) (35.5.0 for Xavier, but check your model) supported L4T into the SBC you can follow the [standard procedure](https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3275/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/quick_start.html#wwpID0E0JD0HA), or, get a fast-forward and simple method using the Jetson Hacks [github repo](https://github.com/jetsonhacks/bootFromExternalStorage) instructions. 

**Important**: The Jetson hacks repo flashes to L4T 5.12 version, if you are using a NVME disk it will be partitioned as a 16GB disk, to use the full space you have to edit the *nvsdkmanager_flash.sh* file to use the flash_l4t_external.xml file instead of the flash_l4t_nvme.xml (lines 136 and 141) [src](https://forums.developer.nvidia.com/t/controlling-app-size-partition-jetpack-5-1-2/265484/3).

![Remember to set Recovery Mode](img/posts/xavier-pins.png Nvidia Xavier covery mode)

After flashing, Ubuntu 20.04 installation will require to set the size of the main partition.

## Things to do after flashing


### Install CUDA packages

Install [CUDA](https://developer.nvidia.com/cuda-12-3-2-download-archive?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=Ubuntu&target_version=20.04&target_type=deb_local) and other ML tooling
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/cuda-ubuntu2004.pinsudo 
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-ubuntu2004-12-3-local_12.3.2-545.23.08-1_arm64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-3-local_12.3.2-545.23.08-1_arm64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt-get install \
 cuda-toolkit-12-3 \
 nvidia-jetpack \
 python3-libnvinfer-dev \
 python2.7-dev \
 python-dev \
 python-py \
 python-attr \
 python-funcsigs \
 python-pluggy \
 python-pytest \
 python-six \
 uff-converter-tf \
 libtbb-dev
```

### Setup Wake-up--on-LAN

Create a [service descriptor](https://necromuralist.github.io/posts/enabling-wake-on-lan/) to restart remotely after suspend the machine (`sudo systemctl suspend`),  in file */etc/systemd/system/wol.service* with the following contents:

```ini
[Unit]
Description=Enable Wake On Lan

[Service]
Type=oneshot
ExecStart = /sbin/ethtool --change eth0 wol g

[Install]
WantedBy=basic.target
```

Set up the service:
```sh
sudo systemctl daemon-reload
sudo systemctl enable wol.service
sudo systemctl status wol
```

### Use Json-stats

Install [jtop](https://github.com/rbonghi/jetson_stats)

```sh
sudo apt install python3-pip
sudo pip3 install -U jetson-stats
sudo reboot now
```

After installing a good config balance for performance of CPU/GPU/Power is the [mode 5](https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-3275/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html) in which you will get 4 cores (1.9 GHz) / 3 GPU TPC (510 MHz) / 10W with *quiet* cooling, you can use *jtop* to set it or the [cli](https://forums.developer.nvidia.com/t/jetson-xavier-nx-how-to-enable-all-cpu/163777):

```sh
sudo nvpmodel -m 5
sudo jetson_clocks
```

In my personal tests using the GPU to serve the Ollama LLMs is required to set the cooling to manual with at least 80% (5051 RPM).

## Running Ollama

Nvidia introduced jetson containers as part of their [cloud-native](https://developer.nvidia.com/embedded/jetson-cloud-native) strategy, it allows to run containers using the GPU (cards and onboard) to accelerate the execution. In jetson the [github repo](https://github.com/dusty-nv/jetson-containers) maintains a series of ML/AI containers compatibles with several L4T kernels.

Start by installing the containers support:
```sh
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
```

Then, you can go and run several tools, in our case the [ollama container](https://github.com/dusty-nv/jetson-containers/tree/master/packages/llm/ollama) as daemon:

```sh
jetson-containers run -d --name ollama dustynv/ollama:r35.4.1
```

which you can set as a systemd service, creating a file */etc/systemd/system/ollama.service* with the following contents:

```ini
[Unit]
Description=Starts Ollama docker server

[Service]
User=nickman
WorkingDirectory=/home/username
Restart=always
RestartSec=10
#Type=oneshot
Environment="ARGS=-d --name ollama dustynv/ollama:r35.4.1"
ExecStart = jetson-containers run $ARGS

[Install]
WantedBy=basic.target
```

And enabling the service:

```sh
sudo systemctl daemon-reload
sudo systemctl enable ollama.service
sudo systemctl status ollama
```

## Demo

Once you have installed all the dependencies, a simple call to [Ollama API](https://editor.swagger.io/?url=https://raw.githubusercontent.com/marscod/ollama/main/api/ollama_api_specification.json) results in a strong consumption of GPU resources:


{{< asciinema key="jtop" >}}

## Conclusion

Despite being an old (6 years?) hardware, the Nvidia Xavier (8 GB) supports the new small LLMs, next time I´ll setup some benchmarks to check speed and quality(?).