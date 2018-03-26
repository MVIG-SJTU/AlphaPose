#include <TH/TH.h>
#include <math.h>

int roi_pooling_forward(int pooled_height, int pooled_width, float spatial_scale,
                        THFloatTensor * features, THFloatTensor * rois, THFloatTensor * output)
{
    // Grab the input tensor
    float * data_flat = THFloatTensor_data(features);
    float * rois_flat = THFloatTensor_data(rois);

    float * output_flat = THFloatTensor_data(output);

    // Number of ROIs
    int num_rois = THFloatTensor_size(rois, 0);
    int size_rois = THFloatTensor_size(rois, 1);
    // batch size
    int batch_size = THFloatTensor_size(features, 0);
    if(batch_size != 1)
    {
        return 0;
    }
    // data height
    int data_height = THFloatTensor_size(features, 1);
    // data width
    int data_width = THFloatTensor_size(features, 2);
    // Number of channels
    int num_channels = THFloatTensor_size(features, 3);

    // Set all element of the output tensor to -inf.
    THFloatStorage_fill(THFloatTensor_storage(output), -1);

    // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    int index_roi = 0;
    int index_output = 0;
    int n;
    for (n = 0; n < num_rois; ++n)
    {
        int roi_batch_ind = rois_flat[index_roi + 0];
        int roi_start_w = round(rois_flat[index_roi + 1] * spatial_scale);
        int roi_start_h = round(rois_flat[index_roi + 2] * spatial_scale);
        int roi_end_w = round(rois_flat[index_roi + 3] * spatial_scale);
        int roi_end_h = round(rois_flat[index_roi + 4] * spatial_scale);
        //      CHECK_GE(roi_batch_ind, 0);
        //      CHECK_LT(roi_batch_ind, batch_size);

        int roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);
        int roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1);
        float bin_size_h = (float)(roi_height) / (float)(pooled_height);
        float bin_size_w = (float)(roi_width) / (float)(pooled_width);

        int index_data = roi_batch_ind * data_height * data_width * num_channels;
        const int output_area = pooled_width * pooled_height;

        int c, ph, pw;
        for (ph = 0; ph < pooled_height; ++ph)
        {
            for (pw = 0; pw < pooled_width; ++pw)
            {
                int hstart = (floor((float)(ph) * bin_size_h));
                int wstart = (floor((float)(pw) * bin_size_w));
                int hend = (ceil((float)(ph + 1) * bin_size_h));
                int wend = (ceil((float)(pw + 1) * bin_size_w));

                hstart = fminf(fmaxf(hstart + roi_start_h, 0), data_height);
                hend = fminf(fmaxf(hend + roi_start_h, 0), data_height);
                wstart = fminf(fmaxf(wstart + roi_start_w, 0), data_width);
                wend = fminf(fmaxf(wend + roi_start_w, 0), data_width);

                const int pool_index = index_output + (ph * pooled_width + pw);
                int is_empty = (hend <= hstart) || (wend <= wstart);
                if (is_empty)
                {
                    for (c = 0; c < num_channels * output_area; c += output_area)
                    {
                        output_flat[pool_index + c] = 0;
                    }
                }
                else
                {
                    int h, w, c;
                    for (h = hstart; h < hend; ++h)
                    {
                        for (w = wstart; w < wend; ++w)
                        {
                            for (c = 0; c < num_channels; ++c)
                            {
                                const int index = (h * data_width + w) * num_channels + c;
                                if (data_flat[index_data + index] > output_flat[pool_index + c * output_area])
                                {
                                    output_flat[pool_index + c * output_area] = data_flat[index_data + index];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Increment ROI index
        index_roi += size_rois;
        index_output += pooled_height * pooled_width * num_channels;
    }
    return 1;
}