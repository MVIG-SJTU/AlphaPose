AlphaPose - Output format
====================================



## Contents
1. [Output Format](#output-format)
    1. [Keypoint Ordering](#keypoint-ordering)
    2. [Heatmap Ordering](#heatmap-ordering)


## Output Format
1. By default, we save the results for all images in one json file, which is similar to the [results format](http://cocodataset.org/#format) used by COCO.
    1. `keypoints` contains the body part locations and detection confidence formatted as `x1,y1,c1,x2,y2,c2,...`. `c` is the confidence score in the range [0,1] for MPII dataset and range [0,6] for COCO dataset.
    2. `score` is the confidence score for the whole person, computed by our parametric pose NMS.
```
[
 // for person_1 in image_1
 {
    "image_id" : string, image_1_name, 
    "category_id" : int, 1 for person
    "keypoints" : [x1,y1,c1,...,xk,yk,ck], 
    "score" : float,
 },
 // for person_2 in image_1
 {
    "image_id" : string, image_1_name, 
    "category_id" : int, 1 for person
    "keypoints" : [x1,y1,c1,...,xk,yk,ck], 
    "score" : float,
 },
 ...
 // for persons in image_2
{
    "image_id" : string, image_2_name, 
    "category_id" : int, 1 for person
    "keypoints" : [x1,y1,c1,...,xk,yk,ck], 
    "score" : float,
 },
 ...
]
```

2. If the `--format` flag is set as 'cmu', we will save the results for each image in the format used by CMU-Pose.
```
{
    "version":0.1,
    "bodies":[
        {"joints":[x1,y1,c1,...,xk,yk,ck]},
        {"joints":[x1,y1,c1,...,xk,yk,ck]},
    ]
}
```

3. If the `--format` flag is set as 'open', we will save the results for each image in the format used by OpenPose.
```
{
    "version":0.1,
    "people":[
        {"pose_keypoints_2d":[x1,y1,c1,...,xk,yk,ck]},
        {"pose_keypoints_2d":[x1,y1,c1,...,xk,yk,ck]},
    ]
}
```

### Keypoint Ordering
The default keypoint order is
```
// Result for COCO (17 body parts)
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
// Result for MPII (16 body parts)
    {0,  "RAnkle"},
    {1,  "Rknee"},
    {2,  "RHip"},
    {3,  "LHip"},
    {4,  "LKnee"},
    {5,  "LAnkle"},
    {6,  "Pelv"},
    {7,  "Thrx"},
    {8,  "Neck"},
    {9,  "Head"},
    {10, "RWrist"},
    {11, "RElbow"},
    {12, "RShoulder"},
    {13, "LShoulder"},
    {14, "LElbow"},
    {15, "LWrist"},
```
If the `--format` flag is set to 'cmu' or 'open', the keypoint order is
```
//Result for COCO (18 body parts)
    {0,  "Nose"},
    {1,  "Neck"},
    {2,  "RShoulder"},
    {3,  "RElbow"},
    {4,  "RWrist"},
    {5,  "LShoulder"},
    {6,  "LElbow"},
    {7,  "LWrist"},
    {8,  "RHip"},
    {9,  "RKnee"},
    {10, "RAnkle"},
    {11, "LHip"},
    {12, "LKnee"},
    {13, "LAnkle"},
    {14, "REye"},
    {15, "LEye"},
    {16, "REar"},
    {17, "LEar"},
// Result for MPII (15 body parts)
    {0,  "Head"},
    {1,  "Neck"},
    {2,  "RShoulder"},
    {3,  "RElbow"},
    {4,  "RWrist"},
    {5,  "LShoulder"},
    {6,  "LElbow"},
    {7,  "LWrist"},
    {8,  "RHip"},
    {9,  "RKnee"},
    {10, "RAnkle"},
    {11, "LHip"},
    {12, "LKnee"},
    {13, "LAnkle"},
    {14, "Thrx"},
```

