#ifndef _CUDA_TENSOR_CONV2_R_VALID_CU_
#define _CUDA_TENSOR_CONV2_R_VALID_CU_

#include "cuda_tensor.cuh"
// code for convolution
namespace apex_tensor{
    namespace cuda_tensor{
        namespace __conv2_r_valid{
            /* load matrix into shared memory */
            template<int y_size, int x_size>
            __device__ void __load_mat_shared_1616( float m_shared[y_size][x_size], 
                                                    const GTensor2D m_global, 
                                                    int y_start ,int x_start ){
                for( int y = 0 ; y < y_size ; y += 16 ){
                    for( int x = 0; x < x_size ; x += 16 ){
                        int yy =  y  + threadIdx.y; // consant in warp
                        int xx =  x  + threadIdx.x; // stride = 1
                        
                        if( ( y_start + yy ) < m_global.y_max 
                            && ( x_start + xx ) < m_global.x_max ){
                            // share  write stride = 1
                            // global read  stride = 1 if start is aligned by 16
                            m_shared[ yy ][ xx ] = m_global[ y_start+yy ].elem[ x_start+xx ];
                        }
                    }
                }  
            }
        };
        
        /* 
           note: block_x and block_y are virtual block id,
           they may not equal to blockIdx.x and blockIdx.y
        */
        template<int y_size, int x_size>
        __device__ void __conv2_r_valid_procedure_1616( float &sum,
                                                        int block_y  , int block_x,                                    
                                                        float s_ft[y_size][x_size] ,
                                                        float s_mm[y_size+16][x_size+16],
                                                        int ans_y_max, int ans_x_max,
                                                        const GTensor2D mat,
                                                        const GTensor2D filter ){
            // load filter into shared memory
            __conv2_r_valid::__load_mat_shared_1616<y_size, x_size>
                ( s_ft , filter, 0, 0 ); 
            // load matrix into shared memory
            __conv2_r_valid::__load_mat_shared_1616<y_size+16, x_size+16>
                ( s_mm , mat, (block_y<<4), (block_x<<4) );
            
            __syncthreads();
            
            const int y_idx = (block_y<<4) + threadIdx.y;
            const int x_idx = (block_x<<4) + threadIdx.x;
            
            if( y_idx < ans_y_max && x_idx < ans_x_max  ){
                for( int dy = 0 ; dy < filter.y_max ; dy ++ ){
                    float ss = 0.0f;
                    for( int dx = 0 ; dx < filter.x_max ; dx ++ ){
                        // s_ft[dy,dx] get by broadcast
                        // s_mm get by stride = 1
                        ss += s_mm[ threadIdx.y + dy ][ threadIdx.x + dx ] * s_ft[ dy ][ dx ];
                    }
                    // for better accuracy
                    sum += ss;
                }
            }
        }
        
        template<int y_size, int x_size>
        __device__ void __conv2_r_valid_procedure_1616( float &sum,
                                                        int   h_idx,
                                                        int   block_y, int block_x, 
                                                        float s_ft[y_size][x_size],
                                                        float s_mm[y_size+16][x_size+16],
                                                        int   ans_y_max, int ans_x_max,
                                                        const GTensor3D mat,
                                                        const GTensor4D filter ){        
            
            for( int v = 0 ;  v < filter.h_max ; v ++ ){                
                __conv2_r_valid_procedure_1616<y_size,x_size>
                    ( sum,
                      block_y   ,  block_x, 
                      s_ft, s_mm, 
                      ans_y_max ,  ans_x_max, 
                      mat[v] , filter[v][h_idx] );
                //wait for other threads
                __syncthreads(); 
            }
        }                       
        
        template<int st_m,int y_size,int x_size>
        __global__ void __conv2_r_valid_kernel_1616( int grid_width, 
                                                     GTensor3D ans,                                                   
                                                     const GTensor3D mat,
                                                     const GTensor4D filter,
                                                     const GTensor1D h_bias   ){
            // unzip the block index
            const int block_z = blockIdx.y;
            const int block_y = blockIdx.x / grid_width;
            const int block_x = blockIdx.x % grid_width;

            __shared__ float bias;
            __shared__ float s_ft[y_size][x_size];
            __shared__ float s_mm[y_size+16][x_size+16];
            
            // load the bias from data 
            if( threadIdx.x == 15 && threadIdx.y == 15 ){
                // we use last thread to do the job, since
                // last thread may more likely to be idle
                bias = h_bias.elem[ block_z ];
                // we don't sync threads here, note we may sync it in the latter operaton
            }

            float sum = 0.0f;
            
            __conv2_r_valid_procedure_1616<y_size,x_size>
                ( sum, block_z, block_y, block_x,
                  s_ft, s_mm, ans.y_max, ans.x_max, mat, filter );

            sum += bias;           
            
            const int  y_idx    = (block_y<<4) + threadIdx.y;
            const int  x_idx    = (block_x<<4) + threadIdx.x;            
            if( y_idx < ans.y_max && x_idx < ans.x_max ){
                store_method::__store<st_m>( ans[ block_z ][ y_idx ].elem[ x_idx ] , sum );    
            }   
        }
        
        template<int st_m>
        inline void conv2_r_valid( GTensor3D &ans,
                                   const GTensor3D &mat,
                                   const GTensor4D &filter,
                                   const GTensor1D &h_bias ){
            // only 16,16 block is allowed to support maxpooling
            if( filter.y_max <= 16 && filter.x_max <= 16 ){
                int  grid_height = (ans.y_max+15) >> 4;
                int  grid_width  = (ans.x_max+15) >> 4;           
                dim3 dimBlock( 16, 16, 1 );
                dim3 dimGrid ( grid_width * grid_height ,  filter.z_max , 1 );
                __conv2_r_valid_kernel_1616 <st_m,16,16> <<<dimGrid,dimBlock>>> ( grid_width, ans , mat, filter, h_bias );
            }
            else{
                error("too large filter size");
            }
        }
    };
};

#endif

