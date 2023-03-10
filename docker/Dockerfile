# might be able to run with newer versions depending on host gpu and cuda version
ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# nvidia xd
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 
# video libs for opencv and build tools
# notably, opencv is lacking avc1, so writing some videos may fail
# however, mp4v should be enough for testing, and writing shouldn't be needed for prod
RUN apt update && apt install ffmpeg libsm6 libxext6 libyaml-dev git -y

# need to install cython for build
# pycocotools is a dependency that should be installed in setup but fails, and so is added here
RUN pip install cython pycocotools

# all dependencies, if you want
# pip install six terminaltables scipy opencv-python matplotlib visdom tqdm tensorboardx easydict pyyaml halpecocotools torch>=1.1.0 torchvision>=0.3.0 munkres timm==0.1.20 natsort

# not a dependency but installed by setup, pre-install to prevent dependency errors
RUN pip install git+https://github.com/yanfengliu/cython_bbox.git
