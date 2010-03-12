#ifndef _APEX_TENSOR_INIT_H_
#define _APEX_TENSOR_INIT_H_

#include <cstdio>
#include "apex_tensor.h"

// functions for initialization
namespace apex_tensor{
    // allocate space for given tensor
    void tensor_alloc( Tensor1D &ts );
    void tensor_alloc( Tensor2D &ts );
    void tensor_alloc( Tensor3D &ts );
    void tensor_alloc( Tensor4D &ts );

    // free space for given tensor
    void tensor_free( Tensor1D &ts );
    void tensor_free( Tensor2D &ts );
    void tensor_free( Tensor3D &ts );
    void tensor_free( Tensor4D &ts );

    // fill the tensor with real value
    void tensor_fill( Tensor1D &ts, TENSOR_FLOAT val );
    void tensor_fill( Tensor2D &ts, TENSOR_FLOAT val );
    void tensor_fill( Tensor3D &ts, TENSOR_FLOAT val );
    void tensor_fill( Tensor4D &ts, TENSOR_FLOAT val );
        
    // save tensor to file
    void tensor_save_to_file( FILE *dst, const Tensor1D &ts );
    void tensor_save_to_file( FILE *dst, const Tensor2D &ts );
    void tensor_save_to_file( FILE *dst, const Tensor3D &ts );
    void tensor_save_to_file( FILE *dst, const Tensor4D &ts );      

    // load tensor from file 
    void tensor_load_from_file( FILE *src, Tensor1D &ts );
    void tensor_load_from_file( FILE *src, Tensor2D &ts );
    void tensor_load_from_file( FILE *src, Tensor3D &ts );
    void tensor_load_from_file( FILE *src, Tensor4D &ts );      
    
};

#endif

