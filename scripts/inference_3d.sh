set -x

CONFIG=$1
CKPT=$2
VIDEO=$3
OUTDIR=${4:-"./examples/res"}

python scripts/demo_3d_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --video ${VIDEO} \
    --outdir ${OUTDIR} \
    --detector yolo  --save_img --save_video --pose_track
