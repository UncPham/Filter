<div align="center">

# Apply Filter

</div>

## Description

In filter project, we are developing an application that applies filters to human faces in an image. Specifically, we use YOLO5Face for face detection, ResNet18 for facial landmarks detection, and then apply filters to the faces using Delaunay triangulation and affine transformation.

In this repository, I implement an application that applies filters to faces using YOLO5Face, ResNet18, along with two algorithms: Delaunay triangulation and affine transformation. Subsequently, I deploy it as a web application using the Gradio library.

You can refer to the implementation of other parts here.

- [Train ResNet18 repository for Facial Landmarks Detection task](https://github.com/PAD2003/facial_landmarks.git)
- [Report on the training process of ResNet18 for facial landmarks detection task](https://api.wandb.ai/links/pad_team/dzmjp7e6)
- [Docker image of a web application applying filters to human faces](https://hub.docker.com/r/pad2003/apply_filter_web_application)

## Installation

```bash
# clone project
git clone https://github.com/PAD2003/apply_filter.git
cd facial_landmarks

# create conda environment
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt

```

## Pipeline

1. Checkpoint for YOLO5Face
    
    Download the checkpoints yolo5n-0.5.pt at [here](https://github.com/deepcam-cn/yolov5-face.git) and place it at yolov5_face_master/weights/yolo5n-0.5.pt.
    
2. Run application
    
    ```bash
    python -m apply_filter.app
    ```

    Now, you can access the application at localhost:7000

## Docker

1. Pull docker image from docker hub
    
    ```bash
    docker pull pad2003/apply_filter_web_application:latest
    ```
    
2. Run docker container
    
    ```bash
    docker run -p 7000:7000 pad2003/apply_filter_web_application:latest
    ```
    
    After the container has started, you can access the application at localhost:7000
    

## Results

Below are images of the application interface.

![demo](imgs/demo.png)
