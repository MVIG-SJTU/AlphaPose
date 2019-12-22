AlphaPose - Frequently Asked Question (FAQ)
============================================

## FAQ
1. [Can't open webcan or video file](#Can't-open-webcan-or-video-file)

## FAQ
### Can't open webcam or video file
**Q:** - I can't open the webcam or video file.

**A**: Try re-install `opencv-python` with version >= 3.3.1.11 by
```
pip3 uninstall opencv_python
pip3 install opencv_python --user
```
Many people meet this problem at https://github.com/opencv/opencv/issues/8471. The solution I use is 
```
sudo cp <path to opencv source repo>/build/lib/python3/cv2.cpython-35m-x86_64-linux-gnu.so /usr/local/lib/python3.5/dist-packages/cv2/cv2.cpython-35m-x86_64-linux-gnu.so
```
The idea is to replace the cv2.so library provided by pypi with the one compiled from sources. You can check for more info at https://github.com/opencv/opencv/issues/8471.

### Can't open webcam
**Q:** - I can't open the webcam with the latest `opencv-python`

**A**: Check if your device is valid by
```
ls /dev/video*
```
Usually you can find `video0`, but if you have a device with other index like `video3`, you can run the program by
```
python3 webcam_demo.py --webcam 3 --outdir examples/res --vis
```

### Program crash
**Q1:** - I meet `Killed` when processing heavy task, like large videos or images with crowded persons.

**A**: Your system meets out of cpu memory and kills the program autoly. Please reduce the length of result buffer by setting the `--qsize` flag. By default length, free cpu memory over 70G+ is recommended in heavy task.

**Q2:** - I meet segmentation fault when processing heavy task, like large videos or images with crowded persons.

**A**: The parallelization module `torch.multiprocessing` is prone to shared memory leaks. Its garbage collection mechanism `torch_shm_manager` may cause segmentation fault under long-time heavy load. We found this issue when processing large videos with hundreds of persons. To avoid this issue, you can set `--sp` flag to use multi-thread instead, which sacrifices a little efficiency for more stablity. 
