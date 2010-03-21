#ifndef _APEX_TENSOR_H_
#define _APEX_TENSOR_H_

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "apex_tensor_config.h"

namespace apex_tensor{
    struct CTensor1D;
    struct CTensor2D;
    struct CTensor3D;
    struct CTensor4D;
    struct GTensor1D;
    struct GTensor2D;
    struct GTensor3D;
    struct GTensor4D;

    typedef float TENSOR_FLOAT;
    void init_tensor_engine_cpu( int seed ); 
    void destroy_tensor_engine_cpu(); 
    void init_stream_engine_gpu( int num_stream );
    void destroy_stream_engine_gpu();
    void init_tensor_engine_gpu(); 
    void destroy_tensor_engine_gpu(); 
    
    // this choose macro is defined for convinience 
    // for choosing between CPU and GPU implementation
#ifdef __APEX_TENSOR_USE_GPU__
    typedef GTensor1D TTensor1D;
    typedef GTensor2D TTensor2D;
    typedef GTensor3D TTensor3D;
    typedef GTensor4D TTensor4D;
 
    inline void init_tensor_engine( int seed ){
        init_tensor_engine_gpu();
    }
    inline void destroy_tensor_engine(){
        destroy_tensor_engine_gpu();
    }
    
    inline void init_stream_engine( int num_stream ){
        init_stream_engine_gpu( num_stream );
    }
    inline void destroy_stream_engine(){
        destroy_stream_engine_gpu();
    }
#else

    typedef CTensor1D TTensor1D;
    typedef CTensor2D TTensor2D;
    typedef CTensor3D TTensor3D;
    typedef CTensor4D TTensor4D;

    inline void init_tensor_engine( int seed ){
        init_tensor_engine_cpu( seed );
    }
    inline void destroy_tensor_engine(){
        destroy_tensor_engine_cpu();
    }
    inline void init_stream_engine( int num_stream ){}
    inline void destroy_stream_engine(){}
#endif    
};

#include "apex_tensor_cpu.h"
#include "apex_tensor_gpu.h"

// definitions for inline functions 
#define TT1D CTensor1D
#define TT2D CTensor2D
#define TT3D CTensor3D
#define TT4D CTensor4D
#include "apex_tensor_inline.cpp"
#undef TT1D 
#undef TT2D 
#undef TT3D 
#undef TT4D 


// definitions for inline functions 
#define TT1D GTensor1D
#define TT2D GTensor2D
#define TT3D GTensor3D
#define TT4D GTensor4D
#include "apex_tensor_inline.cpp"
#undef TT1D 
#undef TT2D 
#undef TT3D 
#undef TT4D 


#endif

