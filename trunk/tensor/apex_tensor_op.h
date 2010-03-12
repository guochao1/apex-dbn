#ifndef _APEX_TENSOR_OP_H_
#define _APEX_TENSOR_OP_H_

#include "apex_tensor.h"

namespace apex_tensor{
    // dst = a + b
    void add( Tensor1D &dst, const Tensor1D &a, const Tensor1D &b );
    // dst = a - b
    void sub( Tensor1D &dst, const Tensor1D &a, const Tensor1D &b );
    // dst  = dot( mat, src  ) 
    void dot  ( Tensor1D &dst, const Tensor2D mat, const Tensor1D &src );    
    // dst  = dot( mat.T, src)
    void dot_t( Tensor1D &dst, const Tensor2D mat, const Tensor1D &src );    
    // dst += dot( mat, src  ) 
    void add_dot  ( Tensor1D &dst, const Tensor2D mat, const Tensor1D &src );    
    // dst += dot( mat.T, src)
    void add_dot_t( Tensor1D &dst, const Tensor2D mat, const Tensor1D &src );    
    // dst -= dot( mat, src  ) 
    void sub_dot  ( Tensor1D &dst, const Tensor2D mat, const Tensor1D &src );    
    // dst -= dot( mat.T, src)
    void sub_dot_t( Tensor1D &dst, const Tensor2D mat, const Tensor1D &src );            
};


#endif
