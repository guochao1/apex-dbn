#ifndef _APEX_TENSOR_PRIVATE_H_
#define _APEX_TENSOR_PRIVATE_H_

#include "apex_tensor.h"
// private functions used by tensor
namespace apex_tensor{
    inline size_t num_bytes( Tensor1D ts ){
        return ts.pitch;
    }
    
    inline size_t num_line( Tensor1D ts ){
        return 1;
    }
    
    inline size_t num_header( Tensor1D ts ){
        return sizeof(size_t)*1;
    }
    
    inline size_t num_bytes( Tensor2D ts ){
        return ts.pitch*ts.y_max;
    }

    inline size_t num_line( Tensor2D ts ){
        return ts.y_max;
    }
    
    inline size_t num_header( Tensor2D ts ){
        return sizeof(size_t)*2;
    }
    
    inline size_t num_bytes( Tensor3D ts ){
        return ts.pitch*ts.y_max*ts.z_max;
    }
    
    inline size_t num_line( Tensor3D ts ){
        return ts.y_max*ts.z_max;
    }

    inline size_t num_header( Tensor3D ts ){
        return sizeof(size_t)*3;
    }
    
    inline size_t num_bytes( Tensor4D ts ){
        return ts.pitch*ts.y_max*ts.z_max*ts.h_max;
    }
    
    inline size_t num_line( Tensor4D ts ){
        return ts.y_max*ts.z_max*ts.h_max;
    }
    
    inline size_t num_header( Tensor4D ts ){
        return sizeof(size_t)*4;
    }
    
    
    template<typename T> 
    inline TENSOR_FLOAT *get_line( T &ts, size_t idx ){
        return (TENSOR_FLOAT*)((char*)ts.elem + idx*ts.pitch);
    }

    template<typename T> 
    inline const TENSOR_FLOAT *get_line_const( const T &ts, size_t idx ){
        return (const TENSOR_FLOAT*)((const char*)ts.elem + idx*ts.pitch);
    }

};

#endif
