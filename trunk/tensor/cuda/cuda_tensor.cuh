#ifndef _CUDA_TENSOR_CUH_
#define _CUDA_TENSOR_CUH_

#include "../apex_tensor_gpu.h"

namespace apex_tensor{
    // private functions used to support tensor op 
    namespace cuda_tensor{     
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
        
        template<typename T>
        inline int get_stride( const T & ts ){
            return (int)ts.pitch/sizeof(TENSOR_FLOAT);
        }
    };

    namespace cuda_tensor{
        const int ALIGN_BITS       = 4;
        const int ALIGN_WIDTH      = 1 << ALIGN_BITS;
        const int BASE_THREAD_BITS = 8;
        const int BASE_THREAD_NUM  = 1<<BASE_THREAD_BITS;

        // support for store metohod
        namespace store_method{
            const int SAVE = 0;
            const int ADD  = 1;
            const int SUB  = 2;
            const int MUL  = 3;
            template<int st_method>
            __device__ void __store( float &dst, float src );

            template<>
            __device__ void __store<SAVE>( float &dst, float src ){
                dst = src;
            }
            template<>
            __device__ void __store<ADD>( float &dst, float src ){
                dst += src;
            }
            template<>
            __device__ void __store<SUB>( float &dst, float src ){
                dst -= src;
            }            
            template<>
            __device__ void __store<MUL>( float &dst, float src ){
                dst *= src;
            }            
        };

        namespace map_method_A{
            const int A_MASK  = 1<<5; 
            const int SIGMOID = 0 | A_MASK;
            
            template<int mm>
            __device__ float __map( float src );
			template<>
            __device__ float __map<SIGMOID>( float src ){
                return 1.0f / ( 1 + __expf( -src ));
            }
        };
    };

    namespace cuda_tensor{
        template<int st_m,typename T>
        inline void store( T &ts, float src );  
    };
    
};
#include "cuda_tensor_op.cu"
#endif
