#ifndef _APEX_TENSOR_GPU_CU_
#define _APEX_TENSOR_GPU_CU_

#include "apex_tensor.h"
#include "cuda/cuda_tensor.cuh"

// GPU implementation of tensor functions
namespace apex_tensor{    
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
        
#define APEX_USE_TEMPLATE_STORE(func_name,sm)                           \
        void func_name( GTensor1D &dst, float src ){                    \
            cuda_tensor::store<sm ,GTensor1D>( dst, src );              \
        }                                                               \
        void func_name( GTensor2D &dst, float src ){                    \
            cuda_tensor::store<sm ,GTensor2D>( dst, src );              \
        }                                                               \
        void func_name( GTensor3D &dst, float src ){                    \
            cuda_tensor::store<sm ,GTensor3D>( dst, src );              \
        }                                                               \
        void func_name( GTensor4D &dst, float src ){                    \
            cuda_tensor::store<sm ,GTensor4D>( dst, src );              \
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

    };

    namespace tensor{
        using namespace cuda_tensor;

        APEX_USE_TEMPLATE_STORE( fill   , store_method::SAVE )
        APEX_USE_TEMPLATE_MAP_A( sigmoid, store_method::SAVE, map_method_A::SIGMOID )
		
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
            
        };
    };
};
#endif
