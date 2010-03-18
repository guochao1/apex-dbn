#ifndef _CUDA_TENSOR_CONV2_FULL_CU_
#define _CUDA_TENSOR_CONV2_FULL_CU_

#include "cuda_tensor.cuh"
namespace apex_tensor{
    namespace cuda_tensor{
        namespace __conv2_full{
            // load matrix into shared memory, dim_x = dim_y
            // pad exceeding dimsions with 0
            template<int y_size, int x_size>
            __device__ void __load_mat_shared_pad_1616( float m_shared[y_size][x_size], 
                                                        const GTensor2D m_global, 
                                                        int y_start ,int x_start ){
                for( int y = 0; y < y_size; y += 16 ){
                    for( int x = 0; x < x_size; x +=16 ){
                        int yy =  y + threadIdx.y; // consant in warp
                        int xx =  x + threadIdx.x; // stride = 1
                        int cy =  y_start + yy;
                        int cx =  x_start + xx;
                        if( cy < m_global.y_max && cx < m_global.x_max 
                            && cy >= 0 && cx >= 0   ){
                            // share  write stride = 1
                            // global read  stride = 1 if start is aligned by 16
                            m_shared[ yy ][ xx ] = m_global[ cy ].elem[ cx ];
                        }else{
                            m_shared[ yy ][ xx ] = 0.0f; 
                    }
                    }
            
                }  
                
            }
            
            // reverse load , the dimesion should be dim_x = dim_y, used to reverse the filter 
            template<int y_size,int x_size>
            __device__ void __load_mat_shared_reverse_1616( float m_shared[y_size][x_size], 
                                                            const GTensor2D g_filter ){
                for( int y = 0; y < y_size; y +=16 ){
                    for( int x = 0; x < x_size; x +=16 ){                        
                        int yy =  y + threadIdx.y; // consant in warp
                        int xx =  x + threadIdx.x; // stride = 1
                        
                        if( yy < g_filter.y_max && xx < g_filter.x_max ){
                            // stride = 1 aligned
                            m_shared[ g_filter.y_max - yy - 1 ][ g_filter.x_max - xx - 1 ] = g_filter[ yy ].elem[ xx ];
                        }
                    }
                }
            }  
        };
        
        /* constraint: dim_x = dim_y */    
        template<int y_size, int x_size>
        __device__ void __conv2_full_procedure_1616( float &sum,
                                                     int block_y, int block_x,
                                                     float s_ft[y_size   ][x_size],
                                                     float s_mm[y_size+16][x_size+16],
                                                     int   ans_y_max, int ans_x_max,
                                                     const GTensor2D mat,
                                                     const GTensor2D filter ){
            // load filter into shared memory
            __conv2_full::__load_mat_shared_reverse_1616< y_size, x_size >( s_ft, filter );
            
            // load matrix into shared memory
            const int y_start = (block_y<<4) - filter.y_max + 1;
            const int x_start = (block_x<<4) - filter.x_max + 1;
            __conv2_full::__load_mat_shared_pad_1616< y_size+16, x_size+16 >
                ( s_mm, mat, y_start, x_start );
            
            __syncthreads();
            
            const int y_idx = (block_y<<4) + threadIdx.y;
            const int x_idx = (block_x<<4) + threadIdx.x;
            
            if( y_idx < ans_y_max && x_idx < ans_x_max ){
                for( int dy = 0; dy < filter.y_max; dy ++ ){
                    for( int dx = 0; dx < filter.x_max; dx ++ ){
                        // s_ft[dy,dx] get by broadcast
                        // s_mm get by stride = 1
                        sum += s_mm[ threadIdx.y + dy ][ threadIdx.x + dx ] * s_ft[ dy ][ dx ] ;
                    }
                }
            }
        }
        
        template<int y_size, int x_size>
        __device__ void __conv2_full_procedure_1616( float &sum,
                                                     int v_idx, int block_y, int block_x,
                                                     float s_ft[y_size   ][x_size] ,
                                                     float s_mm[y_size+16][x_size+16] ,
                                                     const GTensor3D ans,
                                                     const GTensor3D mat,
                                                     const GTensor4D filter ){        
            for( int h = 0 ; h < filter.z_max ; h ++ ){
                __conv2_full_procedure_1616<y_size,x_size>
                    ( sum ,
                      block_y, block_x,
                      s_ft , s_mm , ans.y_max, ans.x_max, mat[h] , filter[v_idx][h] ); 
                __syncthreads();
            }
        }

        /* convolution with bias */
        template<int st_m, int y_size, int x_size>
        __global__ void __conv2_full_kernel_1616( int grid_width,
                                                  GTensor3D ans,
                                                  const GTensor3D mat,
                                                  const GTensor4D filter,
                                                  const GTensor1D v_bias ){
            int block_z = blockIdx.y;
            int block_y = blockIdx.x / grid_width;
            int block_x = blockIdx.x % grid_width;
        
            __shared__ float bias;
            __shared__ float s_ft[y_size   ][x_size];
            __shared__ float s_mm[y_size+16][x_size+16];

            //load the bias
            if( threadIdx.y == 15 && threadIdx.x == 15 ){
                // we use last thread because last thread seems more likely to be idle
                // no need to sync because sync will occur in latter procedure
                bias = v_bias.elem[ block_z ];
            }
        
            float sum = 0.0f;
            
            __conv2_full_procedure_1616<y_size,x_size>
                ( sum, block_z, block_y, block_x,
                  s_ft , s_mm , ans, mat, filter );        
            
            sum += bias;
                        
            const int  y_idx = (block_y<<4) + threadIdx.y;
            const int  x_idx = (block_x<<4) + threadIdx.x;
            
            if( y_idx < ans.y_max && x_idx < ans.x_max ){ 
                store_method::__store<st_m>( ans[ block_z ][ y_idx ].elem[ x_idx ] , sum );    
            }
        }

        template<int st_m>
        inline void conv2_full( GTensor3D ans,
                                const GTensor3D mat,
                                const GTensor4D filter,
                                const GTensor1D v_bias  ){
            if( filter.y_max <= 16 && filter.x_max <= 16 ){
                int  grid_height= (ans.y_max+15) >> 4 ;
                int  grid_width = (ans.x_max+15) >> 4;
                // pack 3D grid into 2D
                dim3 dimBlock( 16, 16, 1 );
                dim3 dimGrid ( grid_width*grid_height , filter.h_max );
                
                __conv2_full_kernel_1616<st_m,16,16> <<<dimGrid,dimBlock>>> ( grid_width, ans, mat, filter, v_bias );
                
            }else{                
                error("too large filter size");
            }
        }
        
    };
};
#endif

