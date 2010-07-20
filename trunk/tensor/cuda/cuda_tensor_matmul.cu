#ifndef _CUDA_TENSOR_MATMUL_CU_
#define _CUDA_TENSOR_MATMUL_CU_

#include "cuda_tensor.cuh"
#include "base/cuda_reduce.cuh"

namespace apex_tensor{

    namespace cuda_tensor{

        // naive version of dot product
        template<int st_m,int block_dim_bits>
        __global__ void dot_simple_kernel( __GT1D ans, __GT1D a, __GT2D b ){
            const int y = blockIdx.x;   
            __shared__ float s_rst[1<<block_dim_bits];
            s_rst[threadIdx.x] = 0.0f;
            
            for( int xx = 0 ; xx < a.x_max ; xx += (1<<block_dim_bits) ){
                int x = xx + threadIdx.x;
                // non coleasced read of b
                if( x < a.x_max ){
                    s_rst[ threadIdx.x ] += a[x] * b[x][y];                
                    // no need to sync, each thread use own space
                }
            }
            __syncthreads();
            cuda_reduce::reduce_1D<cuda_reduce::SUM,block_dim_bits>( s_rst );
            __syncthreads();
            if( threadIdx.x == 0 ){
                store_method::__store<st_m>( ans[ y ], s_rst[0] );
            } 
        }
        
        
        template<int st_m>
        inline void dot_simple( GTensor1D &ans,
                                const GTensor1D &a,
                                const GTensor2D &b ){
            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( ans.x_max, 1, 1 );
                            
            dot_simple_kernel<st_m,BASE_THREAD_BITS><<<dimGrid,dimBlock>>>
                ( __GT(ans), __GT(a), __GT(b) );
        }                        

        template<int st_m,int block_dim_bits>
        __global__ void dot_rt_simple_kernel( __GT1D ans, __GT1D a, __GT2D b ){
            const int y = blockIdx.x;   
            __shared__ float s_rst[1<<block_dim_bits];
            s_rst[threadIdx.x] = 0.0f;
            
            for( int xx = 0 ; xx < a.x_max ; xx += (1<<block_dim_bits) ){
                int x = xx + threadIdx.x;
                if( x < a.x_max ){ 
                    s_rst[ threadIdx.x ] += a[x] * b[y][x];                
                    // no need to sync, each thread use own space
                }
            }
            __syncthreads();
            cuda_reduce::reduce_1D<cuda_reduce::SUM,block_dim_bits>( s_rst );
            __syncthreads();
            if( threadIdx.x == 0 ){
                store_method::__store<st_m>( ans[ y ], s_rst[0] );
            } 
        }
        
        
        template<int st_m>
        inline void dot_rt_simple( GTensor1D &ans,
                                   const GTensor1D &a,
                                   const GTensor2D &b ){
            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( ans.x_max, 1, 1 );
                            
            dot_rt_simple_kernel<st_m,BASE_THREAD_BITS><<<dimGrid,dimBlock>>>
                ( __GT(ans), __GT(a), __GT(b) );
        }                        
        
        template<int st_m,int block_dim_bits>
        __global__ void dot_lt_simple_kernel( __GT2D ans, __GT1D a, __GT1D b ){            
            const int y = blockIdx.x;   

            __shared__ float s_scale;
            if( threadIdx.x == 0 ){
                s_scale = a[ y ];
            }
            __syncthreads();

            for( int xx = 0 ; xx < a.x_max ; xx += (1<<block_dim_bits) ){
                int x = xx + threadIdx.x;                
                store_method::__store<st_m>( ans[ y ][ x ], s_scale * b[ x ] );
            }
        }

        template<int st_m>
        inline void dot_lt_simple( GTensor2D &ans,
                                   const GTensor1D &a,
                                   const GTensor1D &b ){
            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( a.x_max, 1, 1 );
                            
            dot_lt_simple_kernel<st_m,BASE_THREAD_BITS><<<dimGrid,dimBlock>>>
                ( __GT(ans), __GT(a), __GT(b) );
        }                        
    };
};

#endif

