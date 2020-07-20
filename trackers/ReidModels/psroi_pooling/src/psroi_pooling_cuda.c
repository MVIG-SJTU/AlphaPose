#include <THC/THC.h>
#include <math.h>
#include "cuda/psroi_pooling_kernel.h"



extern THCState* state;

int psroi_pooling_forward_cuda(int pooled_height, int pooled_width, float spatial_scale, int group_size, int output_dim,THCudaTensor *features, THCudaTensor* rois, THCudaTensor* output, THCudaIntTensor* mappingchannel){
	float* data_in = THCudaTensor_data(state, features);
	float* rois_in = THCudaTensor_data(state, rois);
	float* output_out = THCudaTensor_data(state, output);
	int* mappingchannel_out = THCudaIntTensor_data(state, mappingchannel);
	//Get # of Rois
	int num_rois = THCudaTensor_size(state, rois, 0);
	int size_rois = THCudaTensor_size(state, rois, 1);
	if (size_rois!=5)
	{
		return -1;
	}

	//Get # of batch_size
	int batch_size = THCudaTensor_size(state, features, 0);

	int data_height = THCudaTensor_size(state, features, 2);
	int data_width = THCudaTensor_size(state, features, 3);
	int num_channels = THCudaTensor_size(state, features, 1);

	cudaStream_t stream = THCState_getCurrentStream(state);

	// call the gpu kernel for psroi_pooling
	PSROIPoolForwardLauncher(data_in, spatial_scale, num_rois, data_height, data_width, num_channels, pooled_height, pooled_width,rois_in, group_size, 
	output_dim, output_out, mappingchannel_out,stream);
	return 1;
}


int psroi_pooling_backward_cuda(int pooled_height, int pooled_width, float spatial_scale, int output_dim, 
THCudaTensor* top_grad, THCudaTensor* rois, THCudaTensor* bottom_grad, THCudaIntTensor* mappingchannel)
{
    	float *top_grad_flat = THCudaTensor_data(state, top_grad);
	float *rois_flat = THCudaTensor_data(state, rois);

	float *bottom_grad_flat = THCudaTensor_data(state, bottom_grad);
    	int *mappingchannel_flat = THCudaIntTensor_data(state, mappingchannel);

    	// Number of ROIs
    	int num_rois = THCudaTensor_size(state, rois, 0);
    	int size_rois = THCudaTensor_size(state, rois, 1);
    	if (size_rois != 5)
    	{
        	return -1;
    	}
    	// batch size
    	int batch_size = THCudaTensor_size(state, bottom_grad, 0);

    	// data height
    	int data_height = THCudaTensor_size(state, bottom_grad, 2);
    	// data width
    	int data_width = THCudaTensor_size(state, bottom_grad, 3);
    	// Number of channels
    	int num_channels = THCudaTensor_size(state, bottom_grad, 1);

    	cudaStream_t stream = THCState_getCurrentStream(state);

    	PSROIPoolBackwardLauncher(top_grad_flat, mappingchannel_flat, batch_size, num_rois, spatial_scale, num_channels, data_height, data_width, pooled_width,	      pooled_height, output_dim, bottom_grad_flat, rois_flat, stream);
        return 1;
}
