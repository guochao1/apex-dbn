#ifndef _APEX_TENSOR_GPU_CU_
#define _APEX_TENSOR_GPU_CU_

#define _APEX_GPU_COMPILE_MODE_
#include "apex_tensor.h"

#include "cuda/cuda_tensor.cuh"
#undef _APEX_GPU_COMPILE_MODE_

#if __APEX_TENSOR_USE_CUBLAS__
#include "cublas.h"
#endif

// GPU implementation of tensor functions
namespace apex_tensor{    
    static int engine_ref_counter = 0;
    void init_tensor_engine_gpu( void ){
        if ( engine_ref_counter++ == 0 ){
            int device_count = 0;
            if( cudaGetDeviceCount( &device_count ) != cudaSuccess ){
                cuda_tensor::error("can't get device information about cuda\n");
            }else{
                if( device_count == 0 ){
                    cuda_tensor::error("There is no device supporting CUDA.\n");
                }else{                
                    int dev;                    
                    if( cudaGetDevice( &dev ) != cudaSuccess )
                        cuda_tensor::error("can't get device to run CUDA\n");                    
                    
                    cudaDeviceProp device_prop;
                    cudaGetDeviceProperties( &device_prop, dev );
                    if( device_prop.major == 9999 && device_prop.minor == 9999 )
                        cuda_tensor::error("There is no device availiable supporting CUDA.\n");
                }            
            }            
            cuda_rand::rand_init();
#if __APEX_TENSOR_USE_CUBLAS__
            cublasInit();
#endif
        }
    }
    void destroy_tensor_engine_gpu(){
        if( --engine_ref_counter <= 0 ){
            cuda_rand::rand_destroy();
#if __APEX_TENSOR_USE_CUBLAS__
            cublasInit();
#endif
        }
    }

    void sync_gpu_threads(){
        cudaThreadSynchronize();
    }

    void init_stream_engine_gpu( int num_stream ){
        cuda_async::init_stream_engine( num_stream );
    }
    void destroy_stream_engine_gpu(){
        cuda_async::destroy_stream_engine();
    }
    
    // asynchronize support
    namespace async{
        void set_dependecy( GTensor1D &dst, int stream_id ){
            cuda_async::set_stream_dep( dst, stream_id );
        }
        void set_dependecy( GTensor2D &dst, int stream_id ){
            cuda_async::set_stream_dep( dst, stream_id );
        }
        void set_dependecy( GTensor3D &dst, int stream_id ){
            cuda_async::set_stream_dep( dst, stream_id );
        }
        void set_dependecy( GTensor4D &dst, int stream_id ){
            cuda_async::set_stream_dep( dst, stream_id );
        }
    };
    
    namespace cuda_tensor{
        template<typename T>
        inline void alloc_space_template( T &ts ){
            size_t pitch;
            cudaError_t err = cudaMallocPitch( (void**)&ts.elem, &pitch, ts.x_max*sizeof(TENSOR_FLOAT), num_line(ts) );
            ts.pitch = (unsigned int)pitch;
            if( err != cudaSuccess ){
                error( cudaGetErrorString(err) );
            } 
        }     

        template<typename T>
        inline void free_space_template( T &ts ){
            cudaError_t err = cudaFree( ts.elem );
            if( err != cudaSuccess ){
                error( cudaGetErrorString(err) );
            }
        }    
        
        template<typename TA,typename TB,enum cudaMemcpyKind kind>
        inline void copy_template( TA dst, const TB src ){
            cudaError_t err = cudaMemcpy2D( dst.elem, (size_t)dst.pitch, src.elem, (size_t)src.pitch, dst.x_max*sizeof(TENSOR_FLOAT), num_line(dst), kind );
            if( err != cudaSuccess ){
                error( cudaGetErrorString(err) );
            }   
        }                 
    };
    
    namespace cuda_tensor{
#define APEX_USE_TEMPLATE_A(func_name)                                  \
        void func_name( GTensor1D &dst ){                               \
            func_name##_template( dst );                                \
        }                                                               \
        void func_name( GTensor2D &dst ){                               \
            func_name##_template( dst );                                \
        }                                                               \
        void func_name( GTensor3D &dst ){                               \
            func_name##_template( dst );                                \
        }                                                               \
        void func_name( GTensor4D &dst  ){                              \
            func_name##_template( dst );                                \
        }                                                               \

#define APEX_USE_TEMPLATE_B(func_name,TA,TB,kind)                       \
        void func_name( TA##1D &dst, const TB##1D &src ){               \
            func_name##_template<TA##1D,TB##1D,kind>( dst,src );        \
        }                                                               \
        void func_name( TA##2D &dst, const TB##2D &src ){               \
            func_name##_template<TA##2D,TB##2D,kind>( dst,src );        \
        }                                                               \
        void func_name( TA##3D &dst, const TB##3D &src ){               \
            func_name##_template<TA##3D,TB##3D,kind>( dst,src );        \
        }                                                               \
        void func_name( TA##4D &dst, const TB##4D &src ){               \
            func_name##_template<TA##4D,TB##4D,kind>( dst,src );        \
        }                                                               \

    };

    // interface function
    namespace tensor{
        using namespace cuda_tensor;
        APEX_USE_TEMPLATE_A( alloc_space )
        APEX_USE_TEMPLATE_A( free_space  )
        APEX_USE_TEMPLATE_B( copy, GTensor, CTensor, cudaMemcpyHostToDevice   )
        APEX_USE_TEMPLATE_B( copy, GTensor, GTensor, cudaMemcpyDeviceToDevice )
        APEX_USE_TEMPLATE_B( copy, CTensor, GTensor, cudaMemcpyDeviceToHost   )
    };

    namespace tensor{
        
#define APEX_USE_TEMPLATE_STORE(func_name,map_m,sm)                     \
        void func_name( GTensor1D &dst, float src ){                    \
            cuda_tensor::map_m<sm ,GTensor1D>( dst, src );              \
        }                                                               \
        void func_name( GTensor2D &dst, float src ){                    \
            cuda_tensor::map_m<sm ,GTensor2D>( dst, src );              \
        }                                                               \
        void func_name( GTensor3D &dst, float src ){                    \
            cuda_tensor::map_m<sm ,GTensor3D>( dst, src );              \
        }                                                               \
        void func_name( GTensor4D &dst, float src ){                    \
            cuda_tensor::map_m<sm ,GTensor4D>( dst, src );              \
        }                                                               \

#define APEX_USE_TEMPLATE_MAP_A(func_name,sm,mm)                        \
        void func_name( GTensor1D &dst, const GTensor1D &src ){         \
            cuda_tensor::map_A<sm ,mm,GTensor1D>( dst, src );           \
        }                                                               \
        void func_name( GTensor2D &dst, const GTensor2D &src ){         \
            cuda_tensor::map_A<sm ,mm,GTensor2D>( dst, src );           \
        }                                                               \
        void func_name( GTensor3D &dst, const GTensor3D &src ){         \
            cuda_tensor::map_A<sm ,mm,GTensor3D>( dst, src );           \
        }                                                               \
        void func_name( GTensor4D &dst, const GTensor4D &src ){         \
            cuda_tensor::map_A<sm ,mm,GTensor4D>( dst, src );           \
        }                                                               \

#define APEX_USE_TEMPLATE_MAP_B(func_name,sm,mm)                        \
        void func_name( GTensor1D &dst, const GTensor1D &src, float srcb ){ \
            cuda_tensor::map_B<sm ,mm,GTensor1D>( dst, src, srcb );     \
        }                                                               \
        void func_name( GTensor2D &dst, const GTensor2D &src, float srcb ){ \
            cuda_tensor::map_B<sm ,mm,GTensor2D>( dst, src, srcb );     \
        }                                                               \
        void func_name( GTensor3D &dst, const GTensor3D &src, float srcb ){ \
            cuda_tensor::map_B<sm ,mm,GTensor3D>( dst, src, srcb );     \
        }                                                               \
        void func_name( GTensor4D &dst, const GTensor4D &src, float srcb ){ \
            cuda_tensor::map_B<sm ,mm,GTensor4D>( dst, src, srcb );     \
        }                                                               \

#define APEX_USE_TEMPLATE_MAP_C(func_name,sm,mm)                        \
        void func_name( GTensor1D &dst, const GTensor1D &a, const GTensor1D &b ){ \
            cuda_tensor::map_C<sm ,mm,GTensor1D>( dst, a, b );          \
        }                                                               \
        void func_name( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b ){ \
            cuda_tensor::map_C<sm ,mm,GTensor2D>( dst, a, b );          \
        }                                                               \
        void func_name( GTensor3D &dst, const GTensor3D &a, const GTensor3D &b ){ \
            cuda_tensor::map_C<sm ,mm,GTensor3D>( dst, a, b );          \
        }                                                               \
        void func_name( GTensor4D &dst, const GTensor4D &a, const GTensor4D &b ){ \
            cuda_tensor::map_C<sm ,mm,GTensor4D>( dst, a, b );          \
        }                                                               \

#define APEX_USE_TEMPLATE_MAP_D(func_name,sm,mm)                        \
        void func_name( GTensor1D &dst, const GTensor1D &a, const GTensor1D &b, float sa, float sb ){ \
            cuda_tensor::map_D<sm ,mm,GTensor1D>( dst, a, b, sa, sb );  \
        }                                                               \
        void func_name( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b, float sa, float sb ){ \
            cuda_tensor::map_D<sm ,mm,GTensor2D>( dst, a, b, sa, sb );  \
        }                                                               \
        void func_name( GTensor3D &dst, const GTensor3D &a, const GTensor3D &b, float sa, float sb ){ \
            cuda_tensor::map_D<sm ,mm,GTensor3D>( dst, a, b, sa, sb );  \
        }                                                               \
        void func_name( GTensor4D &dst, const GTensor4D &a, const GTensor4D &b, float sa, float sb ){ \
            cuda_tensor::map_D<sm ,mm,GTensor4D>( dst, a, b, sa, sb );  \
        }                                                               \

#define APEX_USE_TEMPLATE_MAP_S(func_name,s_name,sm)                    \
        void func_name( GTensor1D &dst, const GTensor1D &src ){         \
            cuda_tensor::s_name<sm ,GTensor1D>( dst, src );             \
        }                                                               \
        void func_name( GTensor2D &dst, const GTensor2D &src ){         \
            cuda_tensor::s_name<sm ,GTensor2D>( dst, src );             \
        }                                                               \
        void func_name( GTensor3D &dst, const GTensor3D &src ){         \
            cuda_tensor::s_name<sm ,GTensor3D>( dst, src );             \
        }                                                               \
        void func_name( GTensor4D &dst, const GTensor4D &src ){         \
            cuda_tensor::s_name<sm ,GTensor4D>( dst, src );             \
        }                                                               \

#define APEX_USE_TEMPLATE_MAP_SS(func_name,s_name,sm)                   \
        void func_name( GTensor1D &dst, const GTensor1D &src, float srcb ){ \
            cuda_tensor::s_name<sm ,GTensor1D>( dst, src, srcb );       \
        }                                                               \
        void func_name( GTensor2D &dst, const GTensor2D &src, float srcb ){ \
            cuda_tensor::s_name<sm ,GTensor2D>( dst, src, srcb );       \
        }                                                               \
        void func_name( GTensor3D &dst, const GTensor3D &src, float srcb ){ \
            cuda_tensor::s_name<sm ,GTensor3D>( dst, src, srcb );       \
        }                                                               \
        void func_name( GTensor4D &dst, const GTensor4D &src, float srcb ){ \
            cuda_tensor::s_name<sm ,GTensor4D>( dst, src, srcb );       \
        }                                                               \

    };

    namespace tensor{
        using namespace cuda_tensor;

        APEX_USE_TEMPLATE_STORE( fill   , store, store_method::SAVE )
        APEX_USE_TEMPLATE_STORE( sample_gaussian, sample_gaussian, store_method::SAVE )
        APEX_USE_TEMPLATE_MAP_A( sigmoid, store_method::SAVE, map_method_A::SIGMOID )
        APEX_USE_TEMPLATE_MAP_B( add    , store_method::SAVE, map_method_B::ADD     )
        APEX_USE_TEMPLATE_MAP_B( sub    , store_method::SAVE, map_method_B::SUB     )
        APEX_USE_TEMPLATE_MAP_B( mul    , store_method::SAVE, map_method_B::MUL     )
        APEX_USE_TEMPLATE_MAP_B( sadd__mul, store_method::ADD, map_method_B::MUL    )
        APEX_USE_TEMPLATE_MAP_C( add    , store_method::SAVE, map_method_B::ADD     )
        APEX_USE_TEMPLATE_MAP_C( sub    , store_method::SAVE, map_method_B::SUB     )
        APEX_USE_TEMPLATE_MAP_C( mul    , store_method::SAVE, map_method_B::MUL     )
        APEX_USE_TEMPLATE_MAP_D( scale_add, store_method::SAVE, map_method_D::SCALE_ADD )
        APEX_USE_TEMPLATE_MAP_D( sadd__scale_add, store_method::ADD, map_method_D::SCALE_ADD )
		
        APEX_USE_TEMPLATE_MAP_S ( sample_binary  , sample_binary  , store_method::SAVE )
        APEX_USE_TEMPLATE_MAP_SS( sample_gaussian, sample_gaussian, store_method::SAVE )
    };

#if __APEX_TENSOR_USE_CUBLAS__
    namespace tensor{
        void dot( GTensor1D &ans, const GTensor1D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemv( 'N', b.x_max, b.y_max, 1.0 , b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 0.0 , ans.elem, 1 );
#else
            cublasSgemv( 'N', b.x_max, b.y_max, 1.0f, b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 0.0f, ans.elem, 1 );
#endif
        }
        void sadd__dot( GTensor1D &ans, const GTensor1D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemv( 'N', b.x_max, b.y_max, 1.0 , b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0 , ans.elem, 1 );
#else
            cublasSgemv( 'N', b.x_max, b.y_max, 1.0f, b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0f, ans.elem, 1 );
#endif
        }
        void ssub__dot( GTensor1D &ans, const GTensor1D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemv( 'N', b.x_max, b.y_max, -1.0 , b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0 , ans.elem, 1 );
#else
            cublasSgemv( 'N', b.x_max, b.y_max, -1.0f, b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0f, ans.elem, 1 );
#endif
        }

        void dot( GTensor2D &ans, const GTensor2D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemm( 'N', 'N', 
                         b.x_max, a.y_max, b.y_max, 1.0 , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cublasSgemm( 'N', 'N', 
                         b.x_max, a.y_max, b.y_max, 1.0f , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0f, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        void sadd__dot( GTensor2D &ans, const GTensor2D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemm( 'N', 'N', 
                         b.x_max, a.y_max, b.y_max, 1.0 , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cublasSgemm( 'N', 'N', 
                         b.x_max, a.y_max, b.y_max, 1.0f , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0f, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        void ssub__dot( GTensor2D &ans, const GTensor2D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemm( 'N', 'N', 
                         b.x_max, a.y_max, b.y_max, -1.0 , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cublasSgemm( 'N', 'N', 
                         b.x_max, a.y_max, b.y_max, -1.0f , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0f, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }

        void dot_rt( GTensor1D &ans, const GTensor1D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemv( 'T', b.x_max, b.y_max, 1.0 , b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 0.0 , ans.elem, 1 );
#else
            cublasSgemv( 'T', b.x_max, b.y_max, 1.0f, b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 0.0f, ans.elem, 1 );
#endif
        }
        void sadd__dot_rt( GTensor1D &ans, const GTensor1D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemv( 'N', b.x_max, b.y_max, 1.0 , b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0 , ans.elem, 1 );
#else
            cublasSgemv( 'T', b.x_max, b.y_max, 1.0f, b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0f, ans.elem, 1 );
#endif
        }
        void ssub__dot_rt( GTensor1D &ans, const GTensor1D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemv( 'T', b.x_max, b.y_max, -1.0 , b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0 , ans.elem, 1 );
#else
            cublasSgemv( 'T', b.x_max, b.y_max, -1.0f, b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0f, ans.elem, 1 );
#endif
        }                

        void dot_rt( GTensor2D &ans, const GTensor2D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemm( 'T', 'N', 
                         b.y_max, a.y_max, b.x_max, 1.0 , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cublasSgemm( 'T', 'N', 
                         b.y_max, a.y_max, b.x_max, 1.0f , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0f, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }

        void sadd__dot_rt( GTensor2D &ans, const GTensor2D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemm( 'T', 'N', 
                         b.y_max, a.y_max, b.x_max, 1.0 , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cublasSgemm( 'T', 'N', 
                         b.y_max, a.y_max, b.x_max, 1.0f , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0f, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        void ssub__dot_rt( GTensor2D &ans, const GTensor2D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemm( 'T', 'N', 
                         b.y_max, a.y_max, b.x_max, -1.0 , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cublasSgemm( 'T', 'N', 
                         b.y_max, a.y_max, b.x_max, -1.0f , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0f, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }

        void dot_lt( GTensor2D &ans, const GTensor1D &a, const GTensor1D &b ){
            dot_lt_simple<store_method::SAVE>( ans, a, b );
        }
        void sadd__dot_lt( GTensor2D &ans, const GTensor1D &a, const GTensor1D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDger( b.x_max, a.x_max, 1.0 , b.elem, 1, a.elem, 1, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cublasSger( b.x_max, a.x_max, 1.0 , b.elem, 1, a.elem, 1, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        void ssub__dot_lt( GTensor2D &ans, const GTensor1D &a, const GTensor1D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDger( b.x_max, a.x_max, -1.0 , b.elem, 1, a.elem, 1, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cublasSger( b.x_max, a.x_max, -1.0 , b.elem, 1, a.elem, 1, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }

        void dot_lt( GTensor2D &ans, const GTensor2D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemm( 'N', 'T', 
                         b.x_max, a.x_max, b.y_max, 1.0 , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cublasSgemm( 'N', 'T', 
                         b.x_max, a.x_max, b.y_max, 1.0f , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0f, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        void sadd__dot_lt( GTensor2D &ans, const GTensor2D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemm( 'N', 'T', 
                         b.x_max, a.x_max, b.y_max, 1.0 , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cublasSgemm( 'N', 'T', 
                         b.x_max, a.x_max, b.y_max, 1.0f , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0f, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        void ssub__dot_lt( GTensor2D &ans, const GTensor2D &a, const GTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cublasDgemm( 'N', 'T', 
                         b.x_max, a.x_max, b.y_max, -1.0 , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cublasSgemm( 'N', 'T', 
                         b.x_max, a.x_max, b.y_max, -1.0f , 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0f, ans.elem, ans.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
    };
    
#else
    namespace tensor{
        using namespace cuda_tensor;
        void dot( GTensor1D &ans, const GTensor1D &a, const GTensor2D &b ){
            dot_rec<store_method::SAVE>( ans, a, b );
        }
        void sadd__dot( GTensor1D &ans, const GTensor1D &a, const GTensor2D &b ){
            dot_rec<store_method::ADD>( ans, a, b );
        }
        void ssub__dot( GTensor1D &ans, const GTensor1D &a, const GTensor2D &b ){
            dot_rec<store_method::SUB>( ans, a, b );
        }

        void dot_rt( GTensor1D &ans, const GTensor1D &a, const GTensor2D &b ){
            dot_rt_simple<store_method::SAVE>( ans, a, b );
        }
        void sadd__dot_rt( GTensor1D &ans, const GTensor1D &a, const GTensor2D &b ){
            dot_rt_simple<store_method::ADD>( ans, a, b );
        }
        void ssub__dot_rt( GTensor1D &ans, const GTensor1D &a, const GTensor2D &b ){
            dot_rt_simple<store_method::SUB>( ans, a, b );
        }

        void dot_lt( GTensor2D &ans, const GTensor1D &a, const GTensor1D &b ){
            dot_lt_simple<store_method::SAVE>( ans, a, b );
        }
        void sadd__dot_lt( GTensor2D &ans, const GTensor1D &a, const GTensor1D &b ){
            dot_lt_simple<store_method::ADD>( ans, a, b );
        }
        void ssub__dot_lt( GTensor2D &ans, const GTensor1D &a, const GTensor1D &b ){
            dot_lt_simple<store_method::SUB>( ans, a, b );
        }
    };
#endif


    namespace tensor{
        using namespace cuda_tensor;
        APEX_USE_TEMPLATE_MAP_C( sadd__abs_err       , store_method::ADD, map_method_B::ABS_ERR )
        APEX_USE_TEMPLATE_MAP_C( sadd__abs_err_rel   , store_method::ADD, map_method_B::ABS_ERR_REL   )
        APEX_USE_TEMPLATE_MAP_C( sadd__abs_err_relT  , store_method::ADD, map_method_B::ABS_ERR_RELT  )        
    };
    
    // support for CRBM
    namespace tensor{
        namespace rbm{
            APEX_USE_TEMPLATE_MAP_A( mean_recified_linear, store_method::SAVE, map_method_A::RELU )
            APEX_USE_TEMPLATE_MAP_S ( sample_recified_linear  , sample_recified_linear  , store_method::SAVE )
        };
        namespace crbm{
            using namespace cuda_tensor;
            void copy_fit( GTensor2D &ans, const CTensor2D &src ){
                    copy_template<GTensor2D,CTensor2D,cudaMemcpyHostToDevice>( ans, src );
            } 
            void copy_fit( GTensor3D &ans, const CTensor3D &src ){
                for( int i = 0 ; i < ans.z_max ; i ++ )
                    copy_template<GTensor2D,CTensor2D,cudaMemcpyHostToDevice>( ans[i], src[i] );
            } 
            void copy_fit( GTensor3D &ans, const GTensor3D &src ){
                for( int i = 0 ; i < ans.z_max ; i ++ )
                    copy_template<GTensor2D,GTensor2D,cudaMemcpyDeviceToDevice>( ans[i], src[i] );
            } 
            void copy_fit( CTensor3D &ans, const GTensor3D &src ){
                for( int i = 0 ; i < ans.z_max ; i ++ )
                    copy_template<CTensor2D,GTensor2D,cudaMemcpyDeviceToHost>( ans[i], src[i] );
            } 
            
            void sample_maxpooling_2D( GTensor3D &state, const GTensor3D &mean, int pool_size ){
                cuda_tensor::sample_maxpooling<store_method::SAVE>( state, mean, pool_size );
            }
            
            void norm_maxpooling_2D( GTensor3D &mean, const GTensor3D &energy, int pool_size ){
                cuda_tensor::norm_maxpooling<store_method::SAVE>( mean, energy, pool_size );
            }
            
            void conv2_r_valid( GTensor3D &ans, const GTensor3D &a, const GTensor4D &filter, const GTensor1D &bias ){
                cuda_tensor::conv2_r_valid<store_method::SAVE>( ans, a, filter, bias );
            }
            
            void conv2_full   ( GTensor3D &ans, const GTensor3D &a, const GTensor4D &filter, const GTensor1D &bias ){
                cuda_tensor::conv2_full<store_method::SAVE>( ans, a, filter, bias );
            }

            void sadd__conv2_r_big_filter( GTensor4D &ans, const GTensor3D &a, const GTensor3D &filter ){
                cuda_tensor::conv2_r_big_filter<store_method::ADD>( ans, a, filter );
            }

            void ssub__conv2_r_big_filter( GTensor4D &ans, const GTensor3D &a, const GTensor3D &filter ){
                cuda_tensor::conv2_r_big_filter<store_method::SUB>( ans, a, filter );
            }
            
            void sadd__sum_2D( GTensor1D &ans, const GTensor3D &src ){
                cuda_tensor::tensor_sum_2D<store_method::ADD,map_method_A::IDENTITY>( ans, src );
            }

            void ssub__sum_2D( GTensor1D &ans, const GTensor3D &src ){
                cuda_tensor::tensor_sum_2D<store_method::SUB,map_method_A::IDENTITY>( ans, src );
            }
            
            void sum_2D( GTensor2D &ans, const GTensor4D &src ){
                cuda_tensor::tensor_sum_2D<store_method::SAVE,map_method_A::IDENTITY>( ans, src );
            }

            void sadd__scale( GTensor4D &ans, const GTensor2D &src, TENSOR_FLOAT scale_src ){
                cuda_tensor::map_E<store_method::ADD,map_method_B::MUL>( ans, src, scale_src );
            }
             
            void refill_edge_area( GTensor3D &ans, const GTensor3D &src, int edge_y_len, int edge_x_len ){
                cuda_tensor::map_A_edge<store_method::SAVE,map_method_A::IDENTITY>( ans, src, edge_y_len, edge_x_len );
            }

            void pool_up( GTensor3D &ans , const GTensor3D &src, int pool_size ){
                cuda_tensor::pool_up<store_method::SAVE,map_method_A::IDENTITY>( ans, src, pool_size );
            }
            
            void add_sparse_info( GTensor1D &sum_mf, GTensor1D &sum_mf_grad, const GTensor3D &src, int pool_size ){
                cuda_tensor::pool_sum<store_method::ADD,map_method_A::IDENTITY,map_method_A::SIGMOID_GRAD,true>
                    ( sum_mf, sum_mf_grad, src, pool_size );
            }

        };
    };
};
#endif
