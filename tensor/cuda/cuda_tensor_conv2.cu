#ifndef _CUDA_TENSOR_CONV2_CU_
#define _CUDA_TENSOR_CONV2_CU_

#include "cuda_tensor.cuh"

namespace apex_tensor{
    namespace cuda_tensor{
        namespace __conv2{
            // load a segment into array, check whether the data is aligned 
            template< int x_size >
            __device__ void __load_line_shared_pad_1616_check_align( float m_shared[x_size],
                                                                     const GTensor1D m_global,
                                                                     int x_start ){
                // noting: x_start may be mis-aligned
                const int x_shift = x_start & 15; // get the shifting area
                if( threadIdx.x >= x_shift ){
                    int cx = x_start + threadIdx.x - x_shift;
                    if( cx < m_global.x_max && cx >= 0 ){
                        m_shared[ threadIdx.x - x_shift ] = m_global.elem[ cx ];
                    }else{
                        m_shared[ threadIdx.x - x_shift ] = 0.0f;
                    }
                }
                for( int x = 16 ; x < x_size ; x += 16 ){
                    int xx = x       + threadIdx.x - x_shift;
                    int cx = x_start + xx;
                    if( cx < m_global.x_max && cx >= 0 ){
                        m_shared[ xx ] = m_global.elem[ cx ]; 
                    }else{
                        m_shared[ xx ] = 0.0f;
                    }
                }                
                if( threadIdx.x < x_shift ){
                    int xx = x_size  + threadIdx.x - x_shift;
                    int cx = x_start + xx;
                    if( cx < m_global.x_max && cx >= 0 ){
                        m_shared[ xx ] = m_global.elem[ cx ];
                    }else{
                        m_shared[ xx ] = 0.0f;
                    }
                }                  
            }
            
            // load data into array, the x_start is ensured to be aligned 
            template< int x_size >
            __device__ void __load_line_shared_pad_1616_aligned( float m_shared[x_size],
                                                                 const GTensor1D m_global,
                                                                 int x_start ){
                for( int x = 0 ; x < x_size ; x += 16 ){
                    int xx = x       + threadIdx.x;
                    int cx = x_start + xx;
                    if( cx < m_global.x_max && cx >= 0 ){
                        m_shared[ xx ] = m_global.elem[ cx ]; 
                    }else{
                        m_shared[ xx ] = 0.0f;
                    }
                }                
            }
            
            // load matrix into shared memory, dim_x = dim_y
            // pad exceeding dimsions with 0
            template<int y_size, int x_size,bool check_align>
            __device__ void __load_mat_shared_pad_1616( float m_shared[y_size][x_size], 
                                                        const GTensor2D m_global, 
                                                        int y_start ,int x_start ){
                for( int y = 0; y < y_size; y += 16 ){
                    int yy =  y + threadIdx.y; // consant in warp
                    int cy =  y_start + yy;
                    if( cy < m_global.y_max && cy >= 0 ){
                        if( check_align ) 
                            __load_line_shared_pad_1616_check_align< x_size >( m_shared[ yy ] , m_global[ cy ] , x_start );
                        else
                            __load_line_shared_pad_1616_aligned< x_size >( m_shared[ yy ] , m_global[ cy ] , x_start );
                    }else{
                        for( int x = 0; x < x_size; x +=16 )
                            m_shared[ yy ][ x + threadIdx.x ] = 0.0f;
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
    };

    // conv2 valid 
    namespace cuda_tensor{
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
            __conv2::__load_mat_shared_pad_1616<y_size, x_size, false>
                ( s_ft , filter, 0, 0 ); 
            // load matrix into shared memory
            __conv2::__load_mat_shared_pad_1616<y_size+16, x_size+16, false>
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

    // conv2_full
    namespace cuda_tensor{
        template<int y_size, int x_size>
        __device__ void __conv2_full_procedure_1616( float &sum,
                                                     int block_y, int block_x,
                                                     float s_ft[y_size   ][x_size],
                                                     float s_mm[y_size+16][x_size+16],
                                                     int   ans_y_max, int ans_x_max,
                                                     const GTensor2D mat,
                                                     const GTensor2D filter ){
            // load filter into shared memory
            __conv2::__load_mat_shared_reverse_1616< y_size, x_size >( s_ft, filter );
            
            // load matrix into shared memory
            const int y_start = (block_y<<4) - filter.y_max + 1;
            const int x_start = (block_x<<4) - filter.x_max + 1;
            __conv2::__load_mat_shared_pad_1616< y_size+16, x_size+16, true >
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
                dim3 dimGrid ( grid_width*grid_height, filter.h_max );
                
                __conv2_full_kernel_1616<st_m,16,16> <<<dimGrid,dimBlock>>> ( grid_width, ans, mat, filter, v_bias );
                
            }else{                
                error("too large filter size");
            }
        }        
    };

    // conv2_r_big_filter , restrict the filter size to be in (16,16)
    namespace cuda_tensor{
        /* calculate a block of convolution */  
        // restrict filter size to be in (16,16)
        __device__ void __conv2_r_big_filter_block_procedure_1616_restricted( float &sum,
                                                                              int   y_start , int x_start,
                                                                              float s_ft [16][16],
                                                                              float s_mat[32][32],
                                                                              const GTensor2D mat, 
                                                                              const GTensor2D filter ){
            // load in file
            __conv2::__load_mat_shared_pad_1616<16,16,false>
                ( s_ft, filter, y_start, x_start );
            // load in matrix 
            __conv2::__load_mat_shared_pad_1616<32,32,false>
                ( s_mat, mat  , y_start, x_start );
            
            __syncthreads();
                                                            
            for( int dy = 0 ; dy < 16 ; dy ++ ){
                for( int dx = 0 ; dx < 16 ; dx ++ ){
                    /* s_ft get by broadcase, mat has no bank conflit  */                         
                    sum += s_ft[ dy ][ dx ] * s_mat[ threadIdx.y + dy ][ threadIdx.x + dx ] ; 
                }
            }        
        }

        template<int st_m>
        __device__ void __conv2_r_big_filter_procedure_1616_restricted( float s_ft [16][16],
                                                                        float s_mat[32][32],
                                                                        GTensor2D ans,
                                                                        const GTensor2D mat, 
                                                                        const GTensor2D filter ){
            float sum = 0.0f;

            for( int yy = 0 ; yy < filter.y_max ; yy += 16 )
                for( int xx = 0 ; xx < filter.x_max ; xx += 16 ){
                    __conv2_r_big_filter_block_procedure_1616_restricted
                        ( sum, yy, xx, s_ft, s_mat, mat, filter );   
                    __syncthreads();
                }
                                                            
            if( threadIdx.y < ans.y_max && threadIdx.x < ans.x_max ){                
                store_method::__store<st_m>( ans[ threadIdx.y ].elem[ threadIdx.x ], sum );
            }            
        }
                
        template<int st_m>
        __global__ void __conv2_r_big_filter_kernel_1616_restricted( GTensor4D ans,
                                                                     const GTensor3D mat, 
                                                                     const GTensor3D filter ){
            __shared__ float s_ft [ 16 ][ 16 ];
            __shared__ float s_mat[ 32 ][ 32 ];
            __conv2_r_big_filter_procedure_1616_restricted<st_m>
                ( s_ft, s_mat, ans[ blockIdx.y ][ blockIdx.x ], mat[ blockIdx.y ], filter[ blockIdx.x ] );
        }

        template<int st_m>
        inline void conv2_r_big_filter( GTensor4D ans,
                                        const GTensor3D mat,
                                        const GTensor3D filter ){
            if( ans.x_max <= 16 && ans.y_max <= 16 ){
                dim3 dimBlock( 16,16, 1 );
                dim3 dimGrid ( ans.z_max, ans.h_max, 1  );
                
                __conv2_r_big_filter_kernel_1616_restricted <st_m><<<dimGrid,dimBlock>>> ( ans, mat, filter );
            }{
                error("too large answer size");
            }
        }        
    };
};
#endif
