sudo docker run -it -d -v $(pwd):/app --gpus all --name training pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
