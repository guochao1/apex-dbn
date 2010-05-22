#ifndef _APEX_TENSOR_CONFIG_H_
#define _APEX_TENSOR_CONFIG_H_
// this file is the configure file of apex tensor
// all the optional configuration of apex_tensor can be configured in this file 

/* configure options */
// architecture of cuda
#ifndef __CUDA_ARCH__ 
#define __CUDA_ARCH__ 200
#endif

// option to open GPU implementation as TTensor
#define __APEX_TENSOR_USE_GPU_IMPL__    

// option to open asynchronize stream support 
#define __APEX_TENSOR_GPU_USE_ASYNC__

// use optimized convolution method for special masks
#define  __CUDA_CONV2_USE_OPT__ 1

// use kahan sum in GPU convolution
//#define __CUDA_CONV2_USE_KAHAN_SUM__

/* debug options */
//option to open debug for cuda_rand 
//#define __DEBUG_CUDA_RAND__   

#endif

