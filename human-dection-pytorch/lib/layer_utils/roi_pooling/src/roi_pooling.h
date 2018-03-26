int roi_pooling_forward(int pooled_height, int pooled_width, float spatial_scale,
                        THFloatTensor * features, THFloatTensor * rois, THFloatTensor * output);