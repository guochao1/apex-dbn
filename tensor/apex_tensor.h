#ifndef _APEX_TENSOR_H_
#define _APEX_TENSOR_H_

#include <cstdio>
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
    
    // inline functions for tensor
    
    // functions defined for tensor
    namespace tensor{
        // allocate space for given tensor
        void alloc_space( Tensor1D &ts );
        void alloc_space( Tensor2D &ts );
        void alloc_space( Tensor3D &ts );
        void alloc_space( Tensor4D &ts );
        
        // free space for given tensor
        void free_space( Tensor1D &ts );
        void free_space( Tensor2D &ts );
        void free_space( Tensor3D &ts );
        void free_space( Tensor4D &ts );
        
        // fill the tensor with real value
        void fill( Tensor1D &ts, TENSOR_FLOAT val );
        void fill( Tensor2D &ts, TENSOR_FLOAT val );
        void fill( Tensor3D &ts, TENSOR_FLOAT val );
        void fill( Tensor4D &ts, TENSOR_FLOAT val );
        
        // save tensor to file
        void save_to_file( const Tensor1D &ts, FILE *dst );
        void save_to_file( const Tensor2D &ts, FILE *dst );
        void save_to_file( const Tensor3D &ts, FILE *dst );
        void save_to_file( const Tensor4D &ts, FILE *dst );      
        
        // load tensor from file 
        void load_from_file( Tensor1D &ts, FILE *src );
        void load_from_file( Tensor2D &ts, FILE *src );
        void load_from_file( Tensor3D &ts, FILE *src );
        void load_from_file( Tensor4D &ts, FILE *src );      
    };    
};

// definitions for inline functions 
#include "apex_tensor_inline.cpp"
#endif


