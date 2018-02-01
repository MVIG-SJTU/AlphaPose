This folder includes Torch code for evaluation and visualization of the single person pose estimation network.


To run this code, the following must be installed:

- [Torch7](https://github.com/torch/torch7)
- hdf5 (and the [torch-hdf5](https://github.com/deepmind/torch-hdf5/) package)
- cudnn
- qlua (for displaying results)

For generating predictions:
`th main-alpha-pose.lua predict ${INPUT_PATH} ${OUTPUT_PATH} ${GPU_NUM} ${BATCH_SIZE} ${DATASET} `

For displaying the demo images:
`qlua main-alpha-pose.lua demo ${INPUT_PATH} ${OUTPUT_PATH} ${GPU_NUM} ${BATCH_SIZE} ${DATASET} `
