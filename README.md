# Object Detection & Classification using Pytorch & YoloV3

Effectively detect objects from a live video stream, video file, or image by utilizing the You Only Look Once (YOLO) Algorithm then classify each object with a COCO library, finally accelerate processing by utilizing PyTorch and CUDA drivers to harness GPU Processing.

Additional Documentation on my Google Docs file located [here](https://docs.google.com/document/d/1PbSLxxjpJVySFz90B8xDVCgukfnXE8T-18ikFa4Im5U/edit?usp=sharing)

Also my half-assed attempt at explaining how the Yolo Algorhitm works [here](https://docs.google.com/document/d/1YhXyALjIhVPoSHYdTMZpPfFjiKQeeyoZsVEdrhWfBfI/edit?usp=sharing)

Followed the guide from Ayoosh Kathuria on how to [Implement YoloV3 with Pytorch from Scratch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/).
His article was very informative and insightful.

Although his code is very raw and unelegant, it performs quite well and after hours of headaches, you're able to break it down.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


#### Prerequisites and Installations

Have an NVIDIA CUDA-Driver capable system. This usually means having some form of NVIDIA Graphics Card installed in your system.

* Python3
* Python3-pip
* Virtualenv 
* Virtualenvwrapper (Optional)
* Video capture source (Webcam)


#### Install Nvidia CUDA Drivers:
```
sudo apt install nvidia-cuda-toolkit
```

#### Install Yolov3 weights & configuration, and COCO Datasets:
Download, rename and relocate necessary weights, datasets, and yolo configuration files into required project directories.

* Relocate yolov3.cfg to projects configuration directory
* Relcoate yolov3.weights to projects weights directory
* Relocate coco.names to projects additional directory


```
wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
wget https://github.com/pjreddie/darknet/blob/master/data/coco.names

```

#### Create a new Virtual Environment
I use virtualenvwrapper for ease of use. If you do not, then you're automatically better at me with virtual environments for going the extra mile. 

```
mkvirtualenv SpecialTopics
```

#### Install Pip3 requirements:

Please note that the PyTorch installation is rather large, you might need to create a temporary temp directory in order to install it. To do so, use:

```
export TMPDIR=/path/to/temporary/temp
```

```
pip3 install -r requirements.txt
```


## Running the program

After these installations and configurations, you should be able to execute the Python program through your Terminal or IDE like so

```
python3 main.py
```
OpenCV will try to capture the first capture device available.


