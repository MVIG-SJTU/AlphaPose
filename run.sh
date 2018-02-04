#!/bin/bash

TEMP=`getopt -o gvliocm:fcds --long gpu:,batch:,video:,list:,indir:,outdir:,format:,mode:,dataset:,vis,sep -n 'wrong args...' -- "$@"`

if [ $? != 0 ] ; then 
    echo "Terminating..." 
    exit 1
fi

eval set -- "${TEMP}"

GPU_ID=0
BATCH_SIZE=2
INPUT_PATH="/"
OUTPUT_PATH="examples/results/"
VIDEO_FILE=""
LIST_FILE=""

WORK_PATH=$(dirname $(readlink -f $0))
MODE="normal"
VIS=false
SEP=false 
DATASET="MPII"
FORMAT="default"

while true ; do
        case "$1" in
                -g|--gpu) GPU_ID="$2" ; shift 2;;
                -b|--batch) BATCH_SIZE="$2" ; shift 2;;
                -v|--video) VIDEO_FILE=${WORK_PATH}/$2 ; shift 2;;
                -l|--list) LIST_FILE=${WORK_PATH}/$2 ; shift 2;;
                -i|--indir) INPUT_PATH=${WORK_PATH}/$2 ; shift 2;;
                -o|--outdir) OUTPUT_PATH=${WORK_PATH}/$2 ; shift 2;;
                -m|--mode) MODE=$2 ; shift 2;;
                -r|--vis) VIS=true ; shift ;;
                -q|--sep) SEP=true ; shift ;;
                -d|--dataset) DATASET=$2 ; shift 2;;
                -f|--format) FORMAT=$2 ; shift 2;;
                --) shift ; break ;;
                *) echo "Internal error!" ; exit 1 ;;
        esac
done

# Get number of gpu numbers
IFS=',' read -ra ARRAY <<< "$GPU_ID"
GPU_NUM=${#ARRAY[@]}

echo ${#VIDEO_FILE}
if [ -n "$VIDEO_FILE" ]; then
    echo "convert video to images..."
    INPUT_PATH=${WORK_PATH}/video-tmp
    if ! [ -e "$INPUT_PATH" ]; then
        mkdir $INPUT_PATH
    fi
    ffmpeg -hide_banner -nostats -loglevel 0 -i ${VIDEO_FILE} -r 10 -f image2 ${INPUT_PATH}"/%05d.jpg"
fi

# echo $INPUT_PATH
# echo $OUTPUT_PATH
# echo $LIST_FILE
# echo $VIDEO_FILE
echo 'generating bbox from Faster RCNN...'

cd ${WORK_PATH}"/human-detection/tools"
CUDA_VISIBLE_DEVICES=${GPU_ID} python demo-alpha-pose.py --inputlist=${LIST_FILE} --inputpath=${INPUT_PATH} --outputpath=${OUTPUT_PATH} --mode=${MODE}

# echo $INPUT_PATH
# echo $OUTPUT_PATH
# echo $LIST_FILE
# echo $VIDEO_FILE

echo 'pose estimation with RMPE...'

cd ${WORK_PATH}"/predict"
if [ "$MODE" = "accurate" ]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} th main-alpha-pose-4crop.lua predict ${INPUT_PATH} ${OUTPUT_PATH} ${GPU_NUM} ${BATCH_SIZE} ${DATASET} 
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} th main-alpha-pose.lua predict ${INPUT_PATH} ${OUTPUT_PATH} ${GPU_NUM} ${BATCH_SIZE} ${DATASET} 
fi

cd ${WORK_PATH}"/predict/json"
if [ "$DATASET" = "COCO" ]; then
    python parametric-pose-nms-COCO.py --outputpath ${OUTPUT_PATH} --sep ${SEP} --format ${FORMAT}
else
    python parametric-pose-nms-MPII.py --outputpath ${OUTPUT_PATH} --sep ${SEP} --format ${FORMAT}
fi

if $VIS; then
    echo 'visualization...'
    if ! [ -e ${OUTPUT_PATH}"/RENDER" ]; then
        mkdir ${OUTPUT_PATH}"/RENDER"
    fi
    python json-video.py --outputpath ${OUTPUT_PATH} --inputpath ${INPUT_PATH}
    if [ -n "$VIDEO_FILE" ]; then
        echo 'rendering video...'
        ffmpeg -r 25 -i ${OUTPUT_PATH}"/RENDER/%05d.jpg" -vcodec libx264 -y -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ${OUTPUT_PATH}"/result_MS.mp4"
    fi
fi

# delete generated video frames
cd ${WORK_PATH}
if [ -n "$VIDEO_FILE" ]; then
    INPUT_PATH=${WORK_PATH}/video-tmp
    rm -rf $INPUT_PATH
fi
