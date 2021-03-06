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

// the precision of tensor
#define __APEX_TENSOR_DOUBLE_PRECISION__ 0

// whether to use BLAS to speed up matrix computation
#define __APEX_TENSOR_USE_BLAS__   1

// use cuBLAS to speed up GPU computation
#define __APEX_TENSOR_USE_CUBLAS__ 1

// use kahan sum in GPU convolution
//#define __CUDA_CONV2_USE_KAHAN_SUM__

/* debug options */
//option to open debug for cuda_rand 
//#define __DEBUG_CUDA_RAND__   

// accuracy of tensor float
namespace apex_tensor{
#if __APEX_TENSOR_DOUBLE_PRECISION__
    typedef double TENSOR_FLOAT;
#else
    typedef float TENSOR_FLOAT;
#endif
};
#endif

