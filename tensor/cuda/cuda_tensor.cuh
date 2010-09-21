#ifndef _CUDA_TENSOR_CUH_
#define _CUDA_TENSOR_CUH_

#include "../apex_tensor_gpu.h"

/*
  definition of global constant used for CUDA 
 */
namespace apex_tensor{    
    /* load unit for memory access */
#if __CUDA_ARCH__>=200
    const int MEM_UNIT_BITS = 5;
#else
    const int MEM_UNIT_BITS = 4;
#endif

    const int MEM_UNIT      = 1 << MEM_UNIT_BITS;
    const int MEM_UNIT_MASK = MEM_UNIT - 1; 

    const int Y_UNIT_BITS   = 4 ;
    const int Y_UNIT        = 1 << Y_UNIT_BITS;
    const int Y_UNIT_MASK   = Y_UNIT - 1; 
};

namespace apex_tensor{
    // support for asynchronize launch
    namespace cuda_async{
        const int NUM_STREAM_MAX = 5;
        int       __num_stream = 0;
        cudaStream_t __arr_stream[ NUM_STREAM_MAX + 1 ];
        
        inline void init_stream_engine( int num_stream ){
#ifdef __APEX_TENSOR_GPU_USE_ASYNC__
            if( __num_stream == 0 ){
                if( num_stream > NUM_STREAM_MAX ) num_stream = NUM_STREAM_MAX;
                __num_stream = num_stream;
                for( int i = 1 ; i <= __num_stream ; i ++ )
                    cudaStreamCreate( &__arr_stream[i] );
            }
#endif
        }
        inline void destroy_stream_engine(){
#ifdef __APEX_TENSOR_GPU_USE_ASYNC__
            for( int i = 1 ; i <= __num_stream ; i ++ )
                cudaStreamDestroy( __arr_stream[i] );
#endif
        }
        
        template<typename T>
        inline void set_stream_dep( T& dst, int stream_id ){
            if( stream_id >= 0 && stream_id <= __num_stream )
                dst.__stream_dep = stream_id;
            else 
                dst.__stream_dep = 0;
        }
        
        // get stream for certain destination
        template<typename T>
        inline cudaStream_t get_stream( T& dst ){
#ifdef __APEX_TENSOR_GPU_USE_ASYNC__            
            if( dst.__stream_dep == 0 ) return 0;
            if( dst.__stream_dep < __num_stream ) return __arr_stream[ dst.__stream_dep ];
#endif
            return 0;
        }        

        template<typename T,typename TA>
        inline cudaStream_t get_stream( T& dst, const TA &src ){
#ifdef __APEX_TENSOR_GPU_USE_ASYNC__            
            if( dst.__stream_dep == 0 ) return 0;
            if( src.__stream_dep != 0 && src.__stream_dep != dst.__stream_dep ) return 0;
            if( dst.__stream_dep < __num_stream ) return __arr_stream[ dst.__stream_dep ];
#endif
            return 0;
        }        

        template<typename T,typename TA>
        inline cudaStream_t get_stream( T &dsta, T &dstb, const TA &src ){
#ifdef __APEX_TENSOR_GPU_USE_ASYNC__            
            if( dsta.__stream_dep == 0 ) return 0;
            if( dsta.__stream_dep != dstb.__stream_dep ) return 0;
            if( src.__stream_dep != 0 && src.__stream_dep != dsta.__stream_dep ) return 0;
            if( dsta.__stream_dep < __num_stream ) return __arr_stream[ dsta.__stream_dep ];
#endif
            return 0;
        }        

        template<typename T,typename TA, typename TB>
        inline cudaStream_t get_stream( T& dst, const TA &srca, const TB &srcb ){
#ifdef __APEX_TENSOR_GPU_USE_ASYNC__            
            if( dst.__stream_dep == 0 ) return 0;
            if( srca.__stream_dep != 0 && srca.__stream_dep != dst.__stream_dep ) return 0;
            if( srcb.__stream_dep != 0 && srcb.__stream_dep != dst.__stream_dep ) return 0;
            if( dst.__stream_dep < __num_stream ) return __arr_stream[ dst.__stream_dep ];
#endif
            return 0;
        }        
    };

    // inner structure of GPU tensor for kernel argument passing 
    namespace cuda_tensor{
        struct __GT1D{
            int    x_max;
            float *elem;          
            inline float &operator[]( int idx ){
                return elem[idx];
            }
            inline const float &operator[]( int idx )const{
                return elem[idx];
            }
        };
        struct __GT2D{
            int    x_max, y_max;
            unsigned int pitch;
            float *elem;  
            inline __GT1D operator[]( int idx ){
                __GT1D a;
                a.elem = (float*)((char*)elem + idx*pitch);
                a.x_max= x_max;
                return a;
            }
            inline const __GT1D operator[]( int idx ) const{
                __GT1D a;
                a.elem = (float*)((char*)elem + idx*pitch);
                a.x_max= x_max;
                return a;
            }
        };
        struct __GT3D{
            int    x_max, y_max, z_max;
            unsigned int pitch;
            float *elem;
            inline __GT2D operator[]( int idx ){
                __GT2D a;
                a.elem = (float*)((char*)elem + idx*pitch*y_max);
                a.x_max= x_max;
                a.y_max= y_max;
                a.pitch= pitch;
                return a;
            }
            inline const __GT2D operator[]( int idx )const{
                __GT2D a;
                a.elem = (float*)((char*)elem + idx*pitch*y_max);
                a.x_max= x_max;
                a.y_max= y_max;
                a.pitch= pitch;
                return a;
            }
        };       
        struct __GT4D{
            int    x_max, y_max, z_max, h_max;
            unsigned int pitch;
            float *elem;
            inline __GT3D operator[]( int idx ){
                __GT3D a;
                a.elem = (float*)((char*)elem + idx*pitch*y_max*z_max);
                a.x_max= x_max;
                a.y_max= y_max;
                a.z_max= z_max;
                a.pitch= pitch;
                return a;
            }
            inline const __GT3D operator[]( int idx )const{
                __GT3D a;
                a.elem = (float*)((char*)elem + idx*pitch*y_max*z_max);
                a.x_max= x_max;
                a.y_max= y_max;
                a.z_max= z_max;
                a.pitch=pitch;
                return a;
            }
        };        
        // constructors 
        inline __GT1D __GT( GTensor1D & ts ){
            __GT1D a;
            a.x_max = ts.x_max;
            a.elem  = ts.elem;
            return a;
        }        
        inline const __GT1D __GT( const GTensor1D & ts ){
            __GT1D a;
            a.x_max = ts.x_max;
            a.elem  = ts.elem;
            return a;
        }
        inline __GT2D __GT( GTensor2D & ts ){
            __GT2D a;
            a.x_max = ts.x_max;
            a.y_max = ts.y_max;
            a.pitch = ts.pitch;
            a.elem  = ts.elem;
            return a;
        }        
        inline const __GT2D __GT( const GTensor2D & ts ){
            __GT2D a;
            a.x_max = ts.x_max;
            a.y_max = ts.y_max;
            a.pitch = ts.pitch;
            a.elem  = ts.elem;
            return a;
        }        
        inline __GT3D __GT( GTensor3D & ts ){
            __GT3D a;
            a.x_max = ts.x_max;
            a.y_max = ts.y_max;
            a.z_max = ts.z_max;
            a.pitch = ts.pitch;
            a.elem  = ts.elem;
            return a;
        }        
        inline const __GT3D __GT( const GTensor3D & ts ){
            __GT3D a;
            a.x_max = ts.x_max;
            a.y_max = ts.y_max;
            a.z_max = ts.z_max;
            a.pitch = ts.pitch;
            a.elem  = ts.elem;
            return a;
        }        
        inline __GT4D __GT( GTensor4D & ts ){
            __GT4D a;
            a.x_max = ts.x_max;
            a.y_max = ts.y_max;
            a.z_max = ts.z_max;
            a.h_max = ts.h_max; 
            a.pitch = ts.pitch;
            a.elem  = ts.elem;
            return a;
        }        
        inline const __GT4D __GT( const GTensor4D & ts ){
            __GT4D a;
            a.x_max = ts.x_max;
            a.y_max = ts.y_max;
            a.z_max = ts.z_max;
            a.h_max = ts.h_max;
            a.pitch = ts.pitch;
            a.elem  = ts.elem;
            return a;
        }                              
    };

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
            return (size_t)ts.pitch;
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
            return (size_t)ts.pitch*ts.y_max;
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
            return (size_t)ts.pitch*ts.y_max*ts.z_max;
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
        inline TENSOR_FLOAT *get_line( T &ts, unsigned int idx ){
            return (TENSOR_FLOAT*)((char*)ts.elem + idx*ts.pitch);
        }
        
        template<typename T> 
        inline const TENSOR_FLOAT *get_line_const( const T &ts, unsigned int idx ){
            return (const TENSOR_FLOAT*)((const char*)ts.elem + idx*ts.pitch);
        }       
    };

    namespace cuda_tensor{
        const int ALIGN_BITS       = MEM_UNIT_BITS;
        const int ALIGN_WIDTH      = 1 << ALIGN_BITS;
        const int BASE_THREAD_BITS = 8;
        const int BASE_THREAD_NUM  = 1 << BASE_THREAD_BITS;
        
        inline __device__ __host__ int get_align_width( int x_max ){
            return ((x_max + ALIGN_WIDTH-1) >> ALIGN_BITS) <<ALIGN_BITS;
        }
        inline __device__ TENSOR_FLOAT *get_line( TENSOR_FLOAT *elem, int idx, unsigned int pitch ){
            return (TENSOR_FLOAT*)( (char*)elem + idx*pitch );
        }
        
        inline __device__ const TENSOR_FLOAT *get_line_const( const TENSOR_FLOAT *elem, int idx, unsigned int pitch ){
            return (const TENSOR_FLOAT*)( (const char*)elem + idx*pitch );
        }       

        // support for store metohod
        namespace store_method{
            const int SAVE = 0;
            const int ADD  = 1;
            const int SUB  = 3;
            const int MUL  = 5;
            
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
            const int A_MASK       = 1<<5; 
            const int IDENTITY     = 0 | A_MASK;
            const int SIGMOID      = 1 | A_MASK;
            const int SIGMOID_GRAD = 2 | A_MASK; 
            const int RELU         = 3 | A_MASK; 
            template<int mm>
            __device__ float __map( float src );
            template<>
            __device__ float __map<IDENTITY>( float src ){
                return src;
            }
            template<>
            __device__ float __map<SIGMOID>( float src ){
                return 1.0f / ( 1.0f + expf( -src ));
            }
            template<>
            __device__ float __map<SIGMOID_GRAD>( float src ){
                return src * ( 1.0f - src );
            }
            template<>
            __device__ float __map<RELU>( float src ){
                if( src > 0 ) 
                    return src + logf( 1.0f + expf( -src ) ); 
                else
                    return logf( 1.0f + expf( src ) ); 
            }
        };

        namespace map_method_B{
            const int B_MASK       = 2<<5; 
            const int ADD          = 0 | B_MASK;
            const int SUB          = 1 | B_MASK;
            const int MUL          = 2 | B_MASK;
            const int ABS_ERR      = 3 | B_MASK;
            const int ABS_ERR_REL  = 4 | B_MASK;
            const int ABS_ERR_RELT = 5 | B_MASK;
            
            
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
            template<>
            __device__ float __map<ABS_ERR>( float a, float b ){
                return fabsf( a - b );
            }
            template<>
            __device__ float __map<ABS_ERR_REL>( float a, float b ){
                return fabsf( 1 - b/a );
            }
            template<>
            __device__ float __map<ABS_ERR_RELT>( float a, float b ){
                return fabsf(a) > 1e-5f ? fabsf( 1 - b/a ): fabsf (a-b)/1e-5f ;
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

    // interface of inline functions
    namespace cuda_tensor{
        template<int st_m,typename T>
        inline void store( T &ts, float src );  
    };
    
};

#include "cuda_tensor_op.cu"
#include "cuda_tensor_sampling.cu"
#include "cuda_tensor_conv2.cu"
#include "cuda_tensor_pooling.cu"
#include "cuda_tensor_matmul.cu"
#endif

