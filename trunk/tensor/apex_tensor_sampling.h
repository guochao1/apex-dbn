#ifndef _APEX_TENSOR_SAMPLING_H_
#define _APEX_TENSOR_SAMPLING_H_

#include <cstdio>
#include "apex_tensor.h"
namespace apex_tensor{
    
    // sample binary distribution
    void sample_binary  ( Tensor1D &state, const Tensor1D &prob );
    
    // sample gaussian distribution with certain sd
    void sample_gaussian( Tensor1D &state, const Tensor1D &mean, float sd );
    
    // sample gaussian distribution with certain mean sd
    void sample_gaussian( Tensor1D &state, float mean, float sd );
    
};
#endif

