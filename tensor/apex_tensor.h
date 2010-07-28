#ifndef _APEX_TENSOR_H_
#define _APEX_TENSOR_H_

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "apex_tensor_config.h"

#ifndef __APEX_TENSOR_USE_GPU__

// macro to switch TTensor from CPU to GPU
#ifdef  __APEX_TENSOR_USE_GPU_IMPL__    
#define __APEX_TENSOR_USE_GPU__    1
#endif

#endif

namespace apex_tensor{    
    struct CTensor1D;
    struct CTensor2D;
    struct CTensor3D;
    struct CTensor4D;
	struct CSTensor1D;
	struct CSTensor2D;
    struct CSparseIndex1D;
    struct CTensor1DSparse;
    struct CSparseIndex2D;
    struct CTensor2DSparse;
  
    struct GTensor1D;
    struct GTensor2D;
    struct GTensor3D;
    struct GTensor4D;
    struct GSparseIndex1D;
    struct GTensor1DSparse;
    struct GSparseIndex2D;
    struct GTensor2DSparse;

    
    void init_tensor_engine_cpu( int seed ); 
    void destroy_tensor_engine_cpu(); 
    void init_stream_engine_gpu( int num_stream );
    void destroy_stream_engine_gpu();
    void init_tensor_engine_gpu(); 
    void destroy_tensor_engine_gpu(); 
    void sync_gpu_threads();

    // this choose macro is defined for convinience 
    // for choosing between CPU and GPU implementation
#if __APEX_TENSOR_USE_GPU__
    typedef GTensor1D TTensor1D;
    typedef GTensor2D TTensor2D;
    typedef GTensor3D TTensor3D;
    typedef GTensor4D TTensor4D;
    typedef GSparseIndex2D   TSparseIndex2D;
    typedef GTensor2DSparse  TTensor2DSparse;
 
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
    inline void sync_threads(){
        sync_gpu_threads();
    }
#else

    typedef CTensor1D  TTensor1D;
    typedef CTensor2D  TTensor2D;
    typedef CTensor3D  TTensor3D;
    typedef CTensor4D  TTensor4D;

    typedef CSTensor1D TSTensor1D;
    typedef CSTensor2D TSTensor2D;
    
    typedef CSparseIndex2D   TSparseIndex2D;
    typedef CTensor2DSparse  TTensor2DSparse;

    inline void init_tensor_engine( int seed ){
        init_tensor_engine_cpu( seed );
    }
    inline void destroy_tensor_engine(){
        destroy_tensor_engine_cpu();
    }
    inline void init_stream_engine( int num_stream ){}
    inline void destroy_stream_engine(){}
    inline void sync_threads(){}
#endif    
};

#include "apex_tensor_cpu.h"
#include "apex_tensor_gpu.h"
#include "apex_tensor_sparse.h"

// definitions for inline functions 
#define TT1D  CTensor1D
#define TT2D  CTensor2D
#define TT3D  CTensor3D
#define TT4D  CTensor4D
#define TT1DS CTensor1DSparse
#define TT2DS CTensor2DSparse
#define TSIDX1D CSparseIndex1D
#define TSIDX2D CSparseIndex2D

#include "apex_tensor_inline.h"
#undef TT1D 
#undef TT2D 
#undef TT3D 
#undef TT4D 
#undef TT1DS 
#undef TT2DS 
#undef TSIDX1D
#undef TSIDX2D

// definitions for inline functions 
#define TT1D  GTensor1D
#define TT2D  GTensor2D
#define TT3D  GTensor3D
#define TT4D  GTensor4D
#define TT1DS GTensor1DSparse
#define TT2DS GTensor2DSparse
#define TSIDX1D GSparseIndex1D
#define TSIDX2D GSparseIndex2D
#include "apex_tensor_inline.h"
#undef TT1D 
#undef TT2D 
#undef TT3D 
#undef TT4D 
#undef TT1DS 
#undef TT2DS
#undef TSIDX1D
#undef TSIDX2D

#endif


