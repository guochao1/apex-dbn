#ifndef _APEX_TENSOR_GPU_CU_
#define _APEX_TENSOR_GPU_CU_

#include "apex_tensor_gpu.h"

// GPU implementation of tensor functions
// TODO 
namespace apex_tensor{    
    // private functions used to support tensor op 
    namespace tensor{     
        inline void check_true( bool exp, const char *s ){
            if( !exp ){
                printf("error:%s\n",s ); exit( -1 );
            }
        }

        inline size_t num_bytes( GTensor1D ts ){
            return ts.pitch;
        }
        
        inline int num_line( GTensor1D ts ){
            return 1;
        }

        inline int num_line( CTensor1D ts ){
            return 1;
        }
        
        inline size_t num_header_bytes( GTensor1D ts ){
            return sizeof(int)*1;
        }
       
        inline int num_elem( GTensor1D ts ){
            return ts.x_max;
        }
        
        inline size_t num_bytes( GTensor2D ts ){
            return ts.pitch*ts.y_max;
        }
        
        inline int num_line( GTensor2D ts ){
            return ts.y_max;
        }
        
        inline int num_line( CTensor2D ts ){
            return ts.y_max;
        }
        
        inline size_t num_header_bytes( GTensor2D ts ){
            return sizeof(int)*2;
        }
        
        inline int num_elem( GTensor2D ts ){
            return ts.x_max * ts.y_max;
        }
        
        inline size_t num_bytes( GTensor3D ts ){
            return ts.pitch*ts.y_max*ts.z_max;
        }
        
        inline int num_line( GTensor3D ts ){
            return ts.y_max*ts.z_max;
        }
        inline int num_line( CTensor3D ts ){
            return ts.y_max*ts.z_max;
        }
        
        inline size_t num_header_bytes( GTensor3D ts ){
            return sizeof(int)*3;
        }
        
        inline int num_elem( GTensor3D ts ){
            return ts.x_max * ts.y_max * ts.z_max;
        }

        inline size_t num_bytes( GTensor4D ts ){
            return ts.pitch*ts.y_max*ts.z_max*ts.h_max;
        }
        
        inline int num_line( GTensor4D ts ){
            return ts.y_max*ts.z_max*ts.h_max;
        }
        inline int num_line( CTensor4D ts ){
            return ts.y_max*ts.z_max*ts.h_max;
        }
        
        inline size_t num_header_bytes( GTensor4D ts ){
            return sizeof(int)*4;
        }
        
        inline int num_elem( GTensor4D ts ){
            return ts.x_max * ts.y_max *ts.z_max * ts.h_max;
        }

        template<typename T>
        inline TENSOR_FLOAT *get_line( T &ts, size_t idx ){
            return get_line( ts.elem, ts.pitch, idx );
        }
                        
        template<typename T> 
        inline const TENSOR_FLOAT *get_line_const( const T &ts, size_t idx ){
            return (const TENSOR_FLOAT*)((const char*)ts.elem + idx*ts.pitch);
        }
    };
    
    namespace tensor{
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
    
    namespace tensor{
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
        APEX_USE_TEMPLATE_A( alloc_space )
        APEX_USE_TEMPLATE_A( free_space  )
        APEX_USE_TEMPLATE_B( copy, GTensor, CTensor, cudaMemcpyHostToDevice   )
        APEX_USE_TEMPLATE_B( copy, GTensor, GTensor, cudaMemcpyDeviceToDevice )
        APEX_USE_TEMPLATE_B( copy, CTensor, GTensor, cudaMemcpyDeviceToHost   )
    };

    // kernels for simple operations
    namespace tensor{
        // fill the tensor with data content
    };

    // support for CRBM
    namespace tensor{
        namespace crbm{
            void copy_fit( GTensor3D &dst, const CTensor3D &src ){
                for( int i = 0 ; i < dst.z_max ; i ++ )
                    copy_template<GTensor2D,CTensor2D,cudaMemcpyHostToDevice>( dst[i], src[i] );
            } 
            
        };
    };
};
#endif
