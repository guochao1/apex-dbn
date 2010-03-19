#ifndef _APEX_TENSOR_GPU_CU_
#define _APEX_TENSOR_GPU_CU_

#include "apex_tensor.h"
#include "cuda/cuda_tensor.cuh"

// GPU implementation of tensor functions
namespace apex_tensor{    
    void init_tensor_engine_gpu( void ){
        cuda_rand::rand_init();
    }
    
    void destroy_tensor_engine_gpu(){
        cuda_rand::rand_destroy();
    }

    namespace cuda_tensor{
        template<typename T>
        inline void alloc_space_template( T &ts ){
            cudaMallocPitch( (void**)&ts.elem, &ts.pitch, ts.x_max*sizeof(TENSOR_FLOAT), num_line(ts) );
        }     

        template<typename T>
        inline void free_space_template( T &ts ){
            cudaFree( ts.elem );
        }    
        
        template<typename TA,typename TB,enum cudaMemcpyKind kind>
        inline void copy_template( TA &dst, const TB &src ){
            cudaMemcpy2D( dst.elem, dst.pitch, src.elem, src.pitch, dst.x_max*sizeof(TENSOR_FLOAT), num_line(dst), kind );   
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
        APEX_USE_TEMPLATE_MAP_C( add    , store_method::SAVE, map_method_B::ADD     )
        APEX_USE_TEMPLATE_MAP_C( sub    , store_method::SAVE, map_method_B::SUB     )
        APEX_USE_TEMPLATE_MAP_C( mul    , store_method::SAVE, map_method_B::MUL     )
        APEX_USE_TEMPLATE_MAP_D( scale_add, store_method::SAVE, map_method_D::SCALE_ADD )
		
        APEX_USE_TEMPLATE_MAP_S ( sample_binary  , sample_binary  , store_method::SAVE )
        APEX_USE_TEMPLATE_MAP_SS( sample_gaussian, sample_gaussian, store_method::SAVE )
    };

    // support for CRBM
    namespace tensor{
        namespace crbm{
            using namespace cuda_tensor;
            void copy_fit( GTensor2D &dst, const CTensor2D &src ){
                    copy_template<GTensor2D,CTensor2D,cudaMemcpyHostToDevice>( dst, src );
            } 
            void copy_fit( GTensor3D &dst, const CTensor3D &src ){
                for( int i = 0 ; i < dst.z_max ; i ++ )
                    copy_template<GTensor2D,CTensor2D,cudaMemcpyHostToDevice>( dst[i], src[i] );
            } 
            void copy_fit( GTensor3D &dst, const GTensor3D &src ){
                for( int i = 0 ; i < dst.z_max ; i ++ )
                    copy_template<GTensor2D,GTensor2D,cudaMemcpyDeviceToDevice>( dst[i], src[i] );
            } 
            
            void sample_maxpooling_2D( GTensor3D &state, const GTensor3D &mean, int pool_size ){
                cuda_tensor::sample_maxpooling<store_method::SAVE>( state, mean, pool_size );
            }
           
            void conv2_r_valid( GTensor3D &dst, const GTensor3D &a, const GTensor4D &filter, const GTensor1D &bias ){
                cuda_tensor::conv2_r_valid<store_method::SAVE>( dst, a, filter, bias );
            }
            
            void conv2_full   ( GTensor3D &dst, const GTensor3D &a, const GTensor4D &filter, const GTensor1D &bias ){
                cuda_tensor::conv2_full<store_method::SAVE>( dst, a, filter, bias );
            }

            void sadd__conv2_r_big_filter( GTensor4D &dst, const GTensor3D &a, const GTensor3D &filter ){
                cuda_tensor::conv2_r_big_filter<store_method::ADD>( dst, a, filter );
            }

            void ssub__conv2_r_big_filter( GTensor4D &dst, const GTensor3D &a, const GTensor3D &filter ){
                cuda_tensor::conv2_r_big_filter<store_method::SUB>( dst, a, filter );
            }

        };
    };
};
#endif
