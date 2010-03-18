#ifndef _CUDA_TENSOR_OP_CU_
#define _CUDA_TENSOR_OP_CU_

#include "cuda_tensor.cuh"
namespace apex_tensor{
    namespace cuda_tensor{
        template<int st_m,int block_dim_bits>
        __global__ void store_kernel( float *elem, int stride,
                                      int y_max  , int x_max ,
                                      float src ){
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;
            const int y   = tid / stride;
            const int x   = tid % stride;

            if( x < x_max  && y < y_max ){
                store_method::__store<st_m>( elem[tid] , src );
            }            
        }

        // store one element to another 
        template<int st_m,typename T>
        inline void store( T &ts, float src ){
            int stride = get_stride( ts );
            int y_max  = num_line( ts );
            int x_max  = ts.x_max;
            
            int num_block = (y_max*stride + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , 1, 1 );
            
            store_kernel<st_m,BASE_THREAD_BITS> <<<dimGrid,dimBlock>>>
                ( ts.elem, stride, y_max, x_max, src );
        }  
        

        template<int st_m,int mapm_A,int block_dim_bits>
        __global__ void map_A_kernel( float *elem_dst, const float *elem_src, 
                                      int stride_dst , int stride_src,
                                      int y_max      , int x_max ){
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;
            const int y   = tid / stride_dst;
            const int x   = tid % stride_dst;
            const int tid_src = y * stride_src + x;  
            
            if( x < x_max  && y < y_max ){
                float val = map_method_A::__map<mapm_A>( elem_src[tid_src] );
                store_method::__store<st_m>( elem_dst[tid] , val );
            }            
        }
        
        template<int st_m,int mapm_A,typename T>
        inline void map_A( T &dst, const T &src ){
            int stride_dst = get_stride( dst );
            int stride_src = get_stride( src );
            int y_max      = num_line( dst );
            int x_max      = dst.x_max;
            
            int num_block = (y_max*stride_dst + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , 1, 1 );
            
            map_A_kernel<st_m,mapm_A,BASE_THREAD_BITS> <<<dimGrid,dimBlock>>>
                ( dst.elem, src.elem, stride_dst,  stride_src, y_max, x_max );
        }  
        
    };
};

#endif
