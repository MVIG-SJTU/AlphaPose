This folder includes Torch code for evaluation and visualization of the single person pose estimation network.


To run this code, the following must be installed:

- [Torch7](https://github.com/torch/torch7)
- hdf5 (and the [torch-hdf5](https://github.com/deepmind/torch-hdf5/) package)
- cudnn
- qlua (for displaying results)

For displaying the demo images:
`qlua main.lua demo`

For generating predictions:
`th main.lua predict-[valid or test]`

For evaluation on a set of validation predictions:
`th main.lua eval` 
