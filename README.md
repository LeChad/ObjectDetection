# Object Detection & Classification using Pytorch & YoloV3

Effectively detect objects from a live video stream, video file, or image by utilizing the You Only Look Once (YOLO) Algorithm then classify each object with a COCO library, finally accelerate processing by utilizing PyTorch and CUDA drivers to harness GPU Processing.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


### Prerequisites and Installations

Have an NVIDIA CUDA-Driver capable system. This usually means having some form of NVIDIA Graphics Card installed in your system.

* Python3
* Python3-pip
* Virtualenv 
* Virtualenvwrapper (Optional)


### Install Nvidia CUDA Drivers:
```
sudo apt install nvidia-cuda-toolkit
```

### Install Yolov3 weights & configuration, and COCO Datasets:
Download, rename and relocate necessary weights, datasets, and yolo configuration files into required project directories.

* Relocate yolov3.cfg to projects configuration directory
* Relcoate yolov3.weights to projects weights directory
* Relocate coco.names to projects additional directory


```
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
wget https://github.com/pjreddie/darknet/blob/master/data/coco.names

```

### Install Pip3 requirements:
Please note that the PyTorch installation is rather large, you might need to create a temporary temp directory in order to install it. 

```
pip3 install -r requirements.txt
```


End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

