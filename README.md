# mx-alphapose
Gluon implementation of AlphaPose.

## Prerequisite

- GluonCV
- MXNet==1.3.0
- tqdm==4.19.1
- Matplotlib
- NumPy

## Usage

Download model weights from [Google Drive](https://drive.google.com/open?id=1TTf8Ox-ECGXRAeX4cHYkEMBDVJEZgBL6) and put it into [sppe/params](sppe/params).

### Image Demo
```bash
MXNET_CPU_WORKER_NTHREADS=2 python3 demo.py
```

### Video Demo
```bash
MXNET_CPU_WORKER_NTHREADS=2 python3 video_demo.py
```
