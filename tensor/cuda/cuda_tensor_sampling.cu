#ifndef _CUDA_TENSOR_SAMPLING_CU_
#define _CUDA_TENSOR_SAMPLING_CU_

#include "rand/cuda_rand.cuh"
#include "rand/cuda_sampling.cuh"

namespace apex_tensor{
    namespace cuda_tensor{        
        template<int st_m,int block_dim_bits>
        __global__ void sample_binary_kernel( float *elem_dst , const float *elem_src, 
                                              size_t pitch_dst, size_t pitch_src,
                                              int y_max       , int x_max,
                                              const float *rnd  ){
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;            
            const int x_mm= get_align_width( x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            
            elem_dst = get_line      ( elem_dst, y, pitch_dst );
            elem_src = get_line_const( elem_src, y, pitch_src );
                        
            if( x < x_max  && y < y_max ){
                float val = cuda_rand::sample_binary( elem_src[x], rnd[tid] - 1.0f );
                store_method::__store<st_m>( elem_dst[x], val );
            }            
        }
        
        template<int st_m,typename T>
        inline void sample_binary( T &dst, const T &src ){
            int stride     = get_align_width( dst.x_max );
            int y_max      = num_line( dst );
            int x_max      = dst.x_max;
            
            int num_block = (y_max*stride + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , 1, 1 );
            const float *rnd = cuda_rand::rand_singles( dimGrid.x * dimBlock.x ); 
            
            sample_binary_kernel<st_m,BASE_THREAD_BITS> <<<dimGrid,dimBlock>>>
                ( dst.elem, src.elem, dst.pitch,  src.pitch, y_max, x_max, rnd );
        } 

        template<int st_m,int block_dim_bits>
        __global__ void sample_gaussian_kernel( float *elem_dst ,
                                                size_t pitch_dst,
                                                int y_max       , int x_max,
                                                const float *rnd, float sd  ){

            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;            
            const float r = cuda_rand::sample_gaussian<block_dim_bits>( rnd[tid] , threadIdx.x ) * sd ;
            const int x_mm= get_align_width( x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            
            elem_dst = get_line( elem_dst, y, pitch_dst );
                        
            if( x < x_max  && y < y_max ){
                store_method::__store<st_m>( elem_dst[x], r );
            }            
        }
        
        template<int st_m,typename T>
        inline void sample_gaussian( T &dst, float sd ){
            int stride     = get_align_width( dst.x_max );
            int y_max      = num_line( dst );
            int x_max      = dst.x_max;
            
            int num_block = (y_max*stride + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , 1, 1 );
            const float *rnd = cuda_rand::rand_singles( dimGrid.x * dimBlock.x ); 
            
            sample_gaussian_kernel<st_m,BASE_THREAD_BITS> <<<dimGrid,dimBlock>>>
                ( dst.elem, dst.pitch, y_max, x_max, rnd, sd );
        } 
        

    };
};
#endif

