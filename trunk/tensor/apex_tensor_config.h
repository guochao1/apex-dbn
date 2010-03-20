#ifndef _APEX_TENSOR_CONFIG_H_
#define _APEX_TENSOR_CONFIG_H_
// this file is the configure file of apex tensor
// all the optional configuration of apex_tensor can be configured in this file 

/* configure options */
 
// option to open GPU implementation as TTensor
#define __APEX_TENSOR_USE_GPU__    

// use kahan sum in GPU convolution
//#define __CUDA_CONV2_USE_KAHAN_SUM__


/* debug options */
//option to open debug for cuda_rand 
//#define __DEBUG_CUDA_RAND__   

#endif

