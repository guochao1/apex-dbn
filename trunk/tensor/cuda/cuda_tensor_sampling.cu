#ifndef _CUDA_TENSOR_SAMPLING_CU_
#define _CUDA_TENSOR_SAMPLING_CU_

#include "rand/cuda_rand.cuh"
#include "rand/cuda_sampling.cuh"

namespace apex_tensor{
    namespace cuda_tensor{        
        // sample binary using prob
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
                float val = cuda_rand::sample_binary( elem_src[x], cuda_rand::get_rand( rnd, tid ) - 1.0f );
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
        
        // sample gaussian with given mean and sd
        template<int st_m,int block_dim_bits>
        __global__ void sample_gaussian_kernel( float *elem_dst , const float *elem_src, 
                                                size_t pitch_dst, size_t pitch_src,
                                                int y_max       , int x_max,
                                                const float *rnd, float sd  ){
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;            
            const float r = cuda_rand::sample_gaussian<block_dim_bits>( cuda_rand::get_rand(rnd,tid)-1.0f , threadIdx.x ) * sd ;
            const int x_mm= get_align_width( x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            
            elem_dst = get_line      ( elem_dst, y, pitch_dst );
            elem_src = get_line_const( elem_src, y, pitch_src );
                        
            if( x < x_max  && y < y_max ){
                store_method::__store<st_m>( elem_dst[x], elem_src[x]+ r );
            }            
        }
        
        template<int st_m,typename T>
        inline void sample_gaussian( T &dst, const T &src, float sd ){
            int stride     = get_align_width( dst.x_max );
            int y_max      = num_line( dst );
            int x_max      = dst.x_max;
            
            int num_block = (y_max*stride + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , 1, 1 );
            const float *rnd = cuda_rand::rand_singles( dimGrid.x * dimBlock.x ); 
            
            sample_gaussian_kernel<st_m,BASE_THREAD_BITS> <<<dimGrid,dimBlock>>>
                ( dst.elem, src.elem, dst.pitch,  src.pitch, y_max, x_max, rnd, sd );
        } 

        // sample gaussian
        template<int st_m,int block_dim_bits>
        __global__ void sample_gaussian_kernel( float *elem_dst ,
                                                size_t pitch_dst,
                                                int y_max       , int x_max,
                                                const float *rnd, float sd  ){

            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;            
            const float r = cuda_rand::sample_gaussian<block_dim_bits>( cuda_rand::get_rand(rnd,tid)-1.0f , threadIdx.x ) * sd ;
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
        
        
        /* 
           sample maxpooling with pool_size = 2^pool_bits
           with block shape < 16 , 16 >
        */
        template<int st_m,int pool_bits>
        __device__ void __sample_maxpooling_procedure_1616( int block_y,
                                                            int block_x,    
                                                            float s_mm[16][16],
                                                            GTensor2D dst,
                                                            const GTensor2D prob,
                                                            const float *rnd ){
            float r = cuda_rand::get_rand( rnd, (threadIdx.y <<4) + threadIdx.x ) - 1.0f;
            
            // load from src 
            for( int y = 0 ; y < (1<<pool_bits) ; y ++ )
                for( int x = 0 ; x < (1<<pool_bits) ; x ++ ){                                
                    int y_idx = block_y * (16 << pool_bits) + (y<<4) + threadIdx.y;
                    int x_idx = block_x * (16 << pool_bits) + (x<<4) + threadIdx.x;
                    
                    // we don't need to sync here since each thread always use the same position 
                    //__syncthreads();
                    
                    // load data into memory 
                    if( y_idx < prob.y_max && x_idx < prob.x_max ) {
                        s_mm[ threadIdx.y ][ threadIdx.x ] = prob[ y_idx ].elem[ x_idx ];
                    }else{
                        s_mm[ threadIdx.y ][ threadIdx.x ] = 0.0f; 
                    }
                    __syncthreads();
                    
                    // if the thread is in this range 
                    if( y == ((threadIdx.y<<pool_bits)>>4) && x == ((threadIdx.x<<pool_bits)>>4) ){
                        // no bank conflict in the same pool, since we only access bank in the same row 
                        cuda_rand::sample_maxpooling<pool_bits>( (threadIdx.y<<pool_bits) &15, 
                                                                 (threadIdx.x<<pool_bits) &15,
                                                                 s_mm, r );                                                 
                    }
                    __syncthreads();
                    
                    if( y_idx < dst.y_max && x_idx < dst.x_max ) {
                        float s = s_mm[ threadIdx.y ][ threadIdx.x ];
                        store_method::__store<st_m>( dst[y_idx].elem[x_idx], s );
                    }
                }
        }
        
        /* pooling kernel, using 3DGrid */
        template<int st_m, int pool_bits>
        __global__ void __sample_maxpooling_1616_kernel_3DGrid( int grid_width, 
                                                                GTensor3D dst, 
                                                                const GTensor3D prob, 
                                                                const float *rnd ){
            const int block_z = blockIdx.y;
            const int block_y = blockIdx.x / grid_width;
            const int block_x = blockIdx.x % grid_width;
            
            __shared__ float s_mm[ 16 ][ 16 ];
            
            __sample_maxpooling_procedure_1616<st_m,pool_bits>
                (  block_y, block_x, s_mm, dst[block_z], prob[block_z], rnd + block_z*(gridDim.x<<8) + (blockIdx.x<<8) );        
        }
        
        /* pooling data up */
        template<int st_m>
        inline void sample_maxpooling( GTensor3D &dst, const GTensor3D &prob, int pool_size ){        
            if( pool_size == 1 ){
                sample_binary<st_m>( dst, prob ); return;
            }
            
            dim3 dimBlock( 16 , 16 );       
            const int d_y_max = (prob.y_max + pool_size-1) / pool_size;  
            const int d_x_max = (prob.x_max + pool_size-1) / pool_size;

            int  grid_height= (d_y_max+15) >> 4 ;        
            int  grid_width = (d_x_max+15) >> 4;

            dim3 dimGrid( grid_width*grid_height, prob.z_max );
        
            const float *rnd  = cuda_rand::rand_singles( (dimGrid.y*dimGrid.x)<<8 );
            
            switch( pool_size ){
            case 2: __sample_maxpooling_1616_kernel_3DGrid<st_m,1><<<dimGrid,dimBlock>>>( grid_width, dst, prob, rnd ); break;
            case 4: __sample_maxpooling_1616_kernel_3DGrid<st_m,2><<<dimGrid,dimBlock>>>( grid_width, dst, prob, rnd ); break;   
            case 8: __sample_maxpooling_1616_kernel_3DGrid<st_m,3><<<dimGrid,dimBlock>>>( grid_width, dst, prob, rnd ); break;   
            default: error("pooling size not supported");
            }
        }                
        
    };
};
#endif

