#ifndef _APEX_TENSOR_H_
#define _APEX_TENSOR_H_

// data structure for tensor
namespace apex_tensor{
    // defines the type of elements in tensor
    typedef float TENSOR_FLOAT;

    struct Tensor1D{

        TENSOR_FLOAT *elem;
        size_t        x_max;        
        Tensor1D(){}
        
        // operators
        inline       TENSOR_FLOAT& operator[]( int idx );
        inline const TENSOR_FLOAT& operator[]( int idx )const;
    };

    struct Tensor2D{
        TENSOR_FLOAT *elem;
        size_t        pitch;
        size_t        x_max, y_max;        

        Tensor2D(){}       

        // operators
        inline       Tensor1D operator[]( int idx );
        inline const Tensor1D operator[]( int idx )const;
    };

    struct Tensor3D{
        TENSOR_FLOAT *elem;
        size_t        pitch;
        size_t        x_max, y_max, z_max;                
        Tensor3D(){}

        // operators
        inline       Tensor2D operator[]( int idx );
        inline const Tensor2D operator[]( int idx )const;
    };
    
    struct Tensor4D{
        TENSOR_FLOAT *elem;
        size_t        pitch;
        size_t        x_max, y_max, z_max, h_max;        

        Tensor4D(){}

        // operators
        inline       Tensor3D operator[]( int idx );
        inline const Tensor3D operator[]( int idx )const;
    };
};

// definitions for inline functions 
#include "apex_tensor_inline.cpp"
#endif


