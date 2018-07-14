AlphaPose - Frequently Asked Question (FAQ)
============================================

## FAQ
1. [Can't open webcan or video file](#Can't-open-webcan-or-video-file)

## FAQ
### Can't open webcan or video file
**Q:** - I can't open webcan or video file.

**A**: Many people meet this problem at https://github.com/opencv/opencv/issues/8471. The solution I use is 
```
sudo cp <path to opencv source repo>/build/lib/python3/cv2.cpython-35m-x86_64-linux-gnu.so /usr/local/lib/python3.5/dist-packages/cv2/cv2.cpython-35m-x86_64-linux-gnu.so
```
The idea is to replace the cv2.so library provided by pypi with the one compiled from sources. You can check for more info at https://github.com/opencv/opencv/issues/8471.
