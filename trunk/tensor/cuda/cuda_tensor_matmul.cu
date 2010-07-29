#ifndef _CUDA_TENSOR_MATMUL_CU_
#define _CUDA_TENSOR_MATMUL_CU_

#include "cuda_tensor.cuh"
#include "base/cuda_reduce.cuh"

namespace apex_tensor{

    namespace cuda_tensor{
        template<int st_m,int xy_dim_bits>
        __global__ void dot_rec_kernel( __GT1D ans, __GT1D a, __GT2D b ){
            const int yy = (blockIdx.x<<xy_dim_bits);    
            const int y  = threadIdx.y + yy;
            __shared__ float s_b[1<<xy_dim_bits][(1<<xy_dim_bits)+1];
            __shared__ float s_rst[1<<xy_dim_bits][1<<xy_dim_bits];

            s_rst[threadIdx.y][threadIdx.x] = 0.0f;
            
            for( int xx = 0 ; xx < a.x_max ; xx += (1<<xy_dim_bits) ){
                const int idx_y = threadIdx.y + xx;
                const int idx_x = threadIdx.x + yy;
                // load into shared buffer
                if( idx_y < b.y_max && threadIdx.x < b.x_max ){
                    s_b[ threadIdx.x ][ threadIdx.y ] = b[ idx_y ] [ idx_x ];
                }
				const int x = threadIdx.x + xx;
				
				__syncthreads();
                if( y < b.x_max && x < b.y_max  ){
                    s_rst[ threadIdx.y ][ threadIdx.x ] += a[ x ] * s_b[ threadIdx.y ][ threadIdx.x ];                   
                }                  
				__syncthreads();
            }

            cuda_reduce::reduce_1D<cuda_reduce::SUM,xy_dim_bits>( s_rst[threadIdx.y] );
            __syncthreads();
            if( threadIdx.x == 0 && y < ans.x_max ){
                store_method::__store<st_m>( ans[ y ], s_rst[threadIdx.y][0] );
            } 
        }
                
        template<int st_m>
        inline void dot_rec( GTensor1D &ans,
                             const GTensor1D &a,
                             const GTensor2D &b ){
            dim3 dimBlock( MEM_UNIT, MEM_UNIT, 1 );
            dim3 dimGrid ( (ans.x_max+MEM_UNIT-1)>>MEM_UNIT_BITS , 1, 1 );
                            
            dot_rec_kernel<st_m,MEM_UNIT_BITS><<<dimGrid,dimBlock>>>
                ( __GT(ans), __GT(a), __GT(b) );
        }                        

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

            for( int xx = 0 ; xx < b.x_max ; xx += (1<<block_dim_bits) ){
                int x = xx + threadIdx.x;                
                if( x < b.x_max ){
                    store_method::__store<st_m>( ans[ y ][ x ], s_scale * b[ x ] );
                }
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

