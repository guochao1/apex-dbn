#ifndef _APEX_TENSOR_INIT_CPP_
#define _APEX_TENSOR_INIT_CPP_

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "apex_tensor.h"
#include "apex_tensor_private.h"

// functions for initialization
namespace apex_tensor{
    template<typename T>
    inline void tensor_alloc_inner( T &ts ){
        ts.pitch= ts.x_max;
        ts.elem = (TENSOR_FLOAT*)malloc( num_bytes( ts ) );
    }
    
    // allocate space for given tensor
    void tensor_alloc( Tensor1D &ts ){
        tensor_alloc_inner( ts );
    }
    void tensor_alloc( Tensor2D &ts ){
        tensor_alloc_inner( ts );
    }
    void tensor_alloc( Tensor3D &ts ){
        tensor_alloc_inner( ts );
    }
    void tensor_alloc( Tensor4D &ts ){
        tensor_alloc_inner( ts );
    }

    // free data
    void tensor_free( Tensor1D &ts ){
        free( ts.elem );
    }
    void tensor_free( Tensor2D &ts ){
        free( ts.elem );
    }
    void tensor_free( Tensor3D &ts ){
        free( ts.elem );
    }
    void tensor_free( Tensor4D &ts ){
        free( ts.elem );
    }

    template<typename T>
    inline void tensor_fill_inner( T &ts, TENSOR_FLOAT val ){
        for( size_t i = 0 ; i < num_line( ts ) ; i ++ ){
            TENSOR_FLOAT *a = get_line( ts, i );
            for( int j = 0 ; j < ts.x_max ; j ++ )
                a[j] = val;
        }
    }
    
    // fill the tensor with real value
    void tensor_fill( Tensor1D &ts,  TENSOR_FLOAT val ){
        tensor_fill_inner( ts , val );
    }
    void tensor_fill( Tensor2D &ts, TENSOR_FLOAT val ){
        tensor_fill_inner( ts , val );
    }
    void tensor_fill( Tensor3D &ts, TENSOR_FLOAT val ){
        tensor_fill_inner( ts , val );
    }
    void tensor_fill( Tensor4D &ts, TENSOR_FLOAT val ){
        tensor_fill_inner( ts , val );
    }    
    
    template<typename T>
    inline void tensor_save_to_file_inner( FILE *dst, const T &ts ){
        fwrite( &ts, num_header_bytes( ts ) , 1 , dst );
        for( size_t i = 0 ; i < num_line( ts ) ; i ++ ){
            TENSOR_FLOAT *a = get_line_const( ts, i );
            fwrite( a, sizeof( TENSOR_FLOAT ) , ts.x_max , dst ); 
        }
    }

    // save tensor to file
    void tensor_save_to_file( FILE *dst, const Tensor1D &ts ){
        tensor_save_to_file( dst, ts );
    }
    
    void tensor_save_to_file( FILE *dst, const Tensor2D &ts ){
        tensor_save_to_file( dst, ts );
    }

    void tensor_save_to_file( FILE *dst, const Tensor3D &ts ){
        tensor_save_to_file( dst, ts );
    }

    void tensor_save_to_file( FILE *dst, const Tensor4D &ts ){
        tensor_save_to_file( dst, ts );
    }
    
    template<typename T>
    inline void tensor_load_from_file_inner( FILE *src, T &ts ){
        fread( &ts, num_header_bytes( ts ) , 1 , src );
        tensor_alloc( ts );
        
        for( size_t i = 0 ; i < num_line( ts ) ; i ++ ){
            TENSOR_FLOAT *a = get_line_const( ts, i );
            fread( a, sizeof( TENSOR_FLOAT ) , ts.x_max , src );        
        }
    }
    
    // save tensor to file
    void tensor_load_from_file( FILE *src, const Tensor1D &ts ){
        tensor_load_from_file( src, ts );
    }
    
    void tensor_load_from_file( FILE *src, const Tensor2D &ts ){
        tensor_load_from_file( src, ts );
    }

    void tensor_load_from_file( FILE *src, const Tensor3D &ts ){
        tensor_load_from_file( src, ts );
    }

    void tensor_load_from_file( FILE *src, const Tensor4D &ts ){
        tensor_load_from_file( src, ts );
    }

};

#endif

