#ifndef _NMS_KERNEL
#define _NMS_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

void _nms(int boxes_num, float * boxes_dev,
          unsigned long long * mask_dev, float nms_overlap_thresh);

#ifdef __cplusplus
}
#endif

#endif

