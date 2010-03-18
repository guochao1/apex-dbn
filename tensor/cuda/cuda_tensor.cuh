#ifndef _CUDA_TENSOR_CUH_
#define _CUDA_TENSOR_CUH_

#include "../apex_tensor_gpu.h"

namespace apex_tensor{
    // private functions used to support tensor op 
    namespace cuda_tensor{
        inline void error( const char *s ){
            printf("error:%s\n",s ); exit( -1 );
        }
        
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
            return (TENSOR_FLOAT*)((char*)ts.elem + idx*ts.pitch);
        }
        
        template<typename T> 
        inline const TENSOR_FLOAT *get_line_const( const T &ts, size_t idx ){
            return (const TENSOR_FLOAT*)((const char*)ts.elem + idx*ts.pitch);
        }       
    };

    namespace cuda_tensor{
        const int ALIGN_BITS       = 4;
        const int ALIGN_WIDTH      = 1 << ALIGN_BITS;
        const int BASE_THREAD_BITS = 8;
        const int BASE_THREAD_NUM  = 1<<BASE_THREAD_BITS;
        
        inline __device__ __host__ int get_align_width( int x_max ){
            return ((x_max + ALIGN_WIDTH-1) >> ALIGN_BITS) <<ALIGN_BITS;
        }
        
        __device__ float *get_line( float *elem, int idx, size_t pitch ){           
            return (float*)((char*)elem + idx*pitch);
        }
        __device__ const float *get_line_const( const float *elem, int idx, size_t pitch ){           
            return (const float*)((const char*)elem + idx*pitch);
        }


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

        namespace map_method_B{
            const int B_MASK  = 2<<5; 
            const int ADD     = 0 | B_MASK;
            const int SUB     = 1 | B_MASK;
            const int MUL     = 2 | B_MASK;
            
            template<int mm>
            __device__ float __map( float a, float b );
			template<>
            __device__ float __map<ADD>( float a, float b ){
                return a + b;
            }
            template<>
            __device__ float __map<SUB>( float a, float b ){
                return a - b;
            }
            template<>
            __device__ float __map<MUL>( float a, float b ){
                return a * b;
            }
        };

        namespace map_method_D{
            const int D_MASK   = 4<<5; 
            const int SCALE_ADD= 0 | D_MASK;
            
            template<int mm>
            __device__ float __map( float a, float b, float sa, float sb );
			template<>
            __device__ float __map<SCALE_ADD>( float a, float b, float sa, float sb ){
                return a * sa + b * sb;
            }            
        };
        
    };

    namespace cuda_tensor{
        template<int st_m,typename T>
        inline void store( T &ts, float src );  
    };
    
};

#include "cuda_tensor_op.cu"
#include "cuda_tensor_sampling.cu"
#include "cuda_tensor_conv2_full.cu"
#include "cuda_tensor_conv2_r_valid.cu"

#endif
