#ifndef _APEX_TENSOR_H_
#define _APEX_TENSOR_H_

// data structure for tensor
namespace apex_tensor{
    // defines the type of elements in tensor
    typedef float TENSOR_FLOAT;

    struct Tensor1D{
        size_t        x_max;        
        size_t        pitch;
        TENSOR_FLOAT *elem;

        Tensor1D(){}
        
        // operators
        inline       TENSOR_FLOAT& operator[]( int idx );
        inline const TENSOR_FLOAT& operator[]( int idx )const;
        inline void operator =  ( TENSOR_FLOAT val );        
        inline void operator += ( const Tensor1D &b );        
    };

    struct Tensor2D{
        size_t        x_max, y_max;        
        size_t        pitch;
        TENSOR_FLOAT *elem;

        Tensor2D(){}       

        // operators
        inline       Tensor1D operator[]( int idx );
        inline const Tensor1D operator[]( int idx )const;
        inline void operator = ( TENSOR_FLOAT val );
        inline void operator +=( const Tensor2D &b );        
    };

    struct Tensor3D{
        size_t        x_max, y_max, z_max;                
        size_t        pitch;
        TENSOR_FLOAT *elem;
        Tensor3D(){}

        // operators
        inline       Tensor2D operator[]( int idx );
        inline const Tensor2D operator[]( int idx )const;
        inline void operator = ( TENSOR_FLOAT val );
        inline void operator +=( const Tensor3D &b );        
    };
    
    struct Tensor4D{
        size_t        x_max, y_max, z_max, h_max;        
        size_t        pitch;

        TENSOR_FLOAT *elem;
        Tensor4D(){}

        // operators
        inline       Tensor3D operator[]( int idx );
        inline const Tensor3D operator[]( int idx )const;
        inline void operator = ( TENSOR_FLOAT val );
        inline void operator +=( const Tensor4D &b );        
    };
};

// definitions for inline functions 
#include "apex_tensor_inline.cpp"
#endif


