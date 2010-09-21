#ifndef _CUDA_TENSOR_SAMPLING_CU_
#define _CUDA_TENSOR_SAMPLING_CU_

#include "cuda_tensor.cuh"
#include "rand/cuda_rand.cuh"
#include "rand/cuda_sampling.cuh"

namespace apex_tensor{
    namespace cuda_tensor{        
        // sample binary using prob
        template<int st_m,int block_dim_bits>
        __global__ void sample_binary_kernel( float *elem_dst , const float *elem_src, 
                                              unsigned int pitch_dst, unsigned int pitch_src,
                                              int y_max       , int x_max,
                                              const float *rnd  ){
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;            
            const int x_mm= get_align_width( x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            
            elem_dst = get_line      ( elem_dst, y, pitch_dst );
            elem_src = get_line_const( elem_src, y, pitch_src );
                        
            if( y < y_max  && x < x_max ){
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

        template<int st_m,int block_dim_bits>
        __global__ void sample_recified_linear_kernel( float *elem_dst , const float *elem_src, 
                                                unsigned int pitch_dst, unsigned int pitch_src,
                                                int y_max       , int x_max,
                                                const float *rnd ){
            __shared__ float s_rnd[ 1<<block_dim_bits ];
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;            
            const float r = cuda_rand::sample_gaussian<block_dim_bits>( cuda_rand::get_rand(rnd,tid), threadIdx.x, s_rnd );
            const int x_mm= get_align_width( x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            
            elem_dst = get_line      ( elem_dst, y, pitch_dst );
            elem_src = get_line_const( elem_src, y, pitch_src );
                        
            if( y < y_max  && x < x_max ){
                float ans = elem_src[x] + r / ( 1.0f + expf( -elem_src[x] ) );
                if( ans < 0.0f ) ans = 0.0f;
                store_method::__store<st_m>( elem_dst[x], ans );
            }            
        }
        
        template<int st_m,typename T>
        inline void sample_recified_linear( T &dst, const T &src ){
            int stride     = get_align_width( dst.x_max );
            int y_max      = num_line( dst );
            int x_max      = dst.x_max;
            
            int num_block = (y_max*stride + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , 1, 1 );
            const float *rnd = cuda_rand::rand_singles( dimGrid.x * dimBlock.x ); 
            
            sample_recified_linear_kernel<st_m,BASE_THREAD_BITS> <<<dimGrid,dimBlock>>>
                ( dst.elem, src.elem, dst.pitch,  src.pitch, y_max, x_max, rnd );
        } 

        // sample gaussian with given mean and sd
        template<int st_m,int block_dim_bits>
        __global__ void sample_gaussian_kernel( float *elem_dst , const float *elem_src, 
                                                unsigned int pitch_dst, unsigned int pitch_src,
                                                int y_max       , int x_max,
                                                const float *rnd, float sd ){
            __shared__ float s_rnd[ 1<<block_dim_bits ];
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;            
            const float r = cuda_rand::sample_gaussian<block_dim_bits>( cuda_rand::get_rand(rnd,tid), threadIdx.x, s_rnd ) * sd;
            const int x_mm= get_align_width( x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            
            elem_dst = get_line      ( elem_dst, y, pitch_dst );
            elem_src = get_line_const( elem_src, y, pitch_src );
                        
            if( y < y_max  && x < x_max ){
                store_method::__store<st_m>( elem_dst[x], elem_src[x] + r );
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
                                                unsigned int pitch_dst,
                                                int y_max       , int x_max,
                                                const float *rnd, float sd  ){
            __shared__ float s_rnd[ 1<<block_dim_bits ];
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;            
            const float r = cuda_rand::sample_gaussian<block_dim_bits>( cuda_rand::get_rand(rnd,tid), threadIdx.x, s_rnd ) * sd;            
            const int x_mm= get_align_width( x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            
            elem_dst = get_line( elem_dst, y, pitch_dst );
                        
            if( y < y_max  && x < x_max ){
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
           with block shape < Y_UNIT , X_UNIT >
        */
        template<int st_m,int pool_bits>
        __device__ void __sample_maxpooling_procedure_rec( int block_y,
                                                           int block_x,    
                                                           float s_mm[Y_UNIT][MEM_UNIT],
                                                           __GT2D dst,
                                                           const __GT2D prob,
                                                           const float *rnd ){
            float r = cuda_rand::get_rand( rnd, (threadIdx.y<<MEM_UNIT_BITS) + threadIdx.x ) - 1.0f;
            
            // load from src 
            for( int y = 0 ; y < (1<<pool_bits) ; y ++ )
                for( int x = 0 ; x < (1<<pool_bits) ; x ++ ){                                
                    int y_idx = block_y * (Y_UNIT   << pool_bits) + (y<<Y_UNIT_BITS)   + threadIdx.y;
                    int x_idx = block_x * (MEM_UNIT << pool_bits) + (x<<MEM_UNIT_BITS) + threadIdx.x;
                    
                    // we don't need to sync here since each thread always use the same position 
                    //__syncthreads();
                    
                    // load data into memory 
                    if( y_idx < prob.y_max && x_idx < prob.x_max ) {
                        s_mm[ threadIdx.y ][ threadIdx.x ] = prob[ y_idx ][ x_idx ];
                    }else{
                        s_mm[ threadIdx.y ][ threadIdx.x ] = 0.0f; 
                    }
                    __syncthreads();
                    
                    // if the thread is in this range 
                    if( y == ((threadIdx.y<<pool_bits)>>Y_UNIT_BITS) && x == ((threadIdx.x<<pool_bits)>>MEM_UNIT_BITS) ){
                        // no bank conflict in the same pool, since we only access bank in the same row 
                        cuda_rand::sample_maxpooling<pool_bits,MEM_UNIT>( (threadIdx.y<<pool_bits) &Y_UNIT_MASK, 
                                                                          (threadIdx.x<<pool_bits) &MEM_UNIT_MASK,
                                                                          s_mm, r );                                                 
                    }
                    __syncthreads();
                    
                    if( y_idx < dst.y_max && x_idx < dst.x_max ) {
                        float s = s_mm[ threadIdx.y ][ threadIdx.x ];
                        store_method::__store<st_m>( dst[y_idx][x_idx], s );
                    }
                }
        }
        
        /* pooling kernel, using 3DGrid */
        template<int st_m, int pool_bits>
        __global__ void __sample_maxpooling_rec_kernel_3DGrid( int grid_width, 
                                                               __GT3D dst, 
                                                               const __GT3D prob, 
                                                               const float *rnd ){
            const int block_z = blockIdx.y;
            const int block_y = blockIdx.x / grid_width;
            const int block_x = blockIdx.x % grid_width;
            
            __shared__ float s_mm[ Y_UNIT ][ MEM_UNIT ];
            
            __sample_maxpooling_procedure_rec<st_m,pool_bits>
                (  block_y, block_x, s_mm, dst[block_z], prob[block_z], rnd + 
                   block_z*(gridDim.x<<(MEM_UNIT_BITS+Y_UNIT_BITS)) + (blockIdx.x<<(MEM_UNIT_BITS+Y_UNIT_BITS)) );        
        }
        
        template<int st_m, int pool_bits>
        inline void __sample_maxpooling_rec( GTensor3D &dst, const GTensor3D &prob ){
            dim3 dimBlock( MEM_UNIT , Y_UNIT );       
            const int d_y_max = (prob.y_max + (1<<pool_bits) - 1) >> pool_bits;  
            const int d_x_max = (prob.x_max + (1<<pool_bits) - 1) >> pool_bits;

            int  grid_height= (d_y_max+Y_UNIT-1  ) >> Y_UNIT_BITS ;        
            int  grid_width = (d_x_max+MEM_UNIT-1) >> MEM_UNIT_BITS;

            dim3 dimGrid( grid_width*grid_height, prob.z_max );
        
            const float *rnd  = cuda_rand::rand_singles( (dimGrid.y*dimGrid.x)<<(MEM_UNIT_BITS+Y_UNIT_BITS) );
            
            __sample_maxpooling_rec_kernel_3DGrid<st_m,pool_bits><<<dimGrid,dimBlock>>>( grid_width, __GT(dst), __GT(prob), rnd );
        }        
                
        /* 
           sample maxpooling with pool_size 
           with block shape < pool_size , 16*pool_size >
        */
        template<int st_m,int pool_size>
        __device__ void __sample_maxpooling_procedure_ord( int block_y,
                                                           int block_x,    
                                                           float s_mm[pool_size][MEM_UNIT*pool_size],
                                                           __GT2D dst,
                                                           const __GT2D prob,
                                                           const float *rnd ){
            float r = cuda_rand::get_rand( rnd, (threadIdx.y*pool_size*MEM_UNIT) + threadIdx.x ) - 1.0f;
            
            // load from src 
            for( int y = 0 ; y < pool_size ; y ++ )
                for( int x = 0 ; x < pool_size ; x ++ ){                                
                    int y_idx = block_y*pool_size*pool_size    + y*pool_size    + threadIdx.y;
                    int x_idx = block_x*pool_size*pool_size*MEM_UNIT + x*pool_size*MEM_UNIT + threadIdx.x;
                    
                    // we don't need to sync here since each thread always use the same position 
                    //__syncthreads();
                    
                    // load data into memory 
                    if( y_idx < prob.y_max && x_idx < prob.x_max ) {
                        s_mm[ threadIdx.y ][ threadIdx.x ] = prob[ y_idx ][ x_idx ];
                    }else{
                        s_mm[ threadIdx.y ][ threadIdx.x ] = 0.0f; 
                    }
                    __syncthreads();
                    
                    // if the thread is in this range 
                    if( y == threadIdx.y && x == (threadIdx.x>>MEM_UNIT_BITS) ){
                        // no bank conflict in the same pool, since we only access bank in the same row 
                        cuda_rand::sample_maxpooling_ord<pool_size,MEM_UNIT>( 0, 
                                                                              (threadIdx.x & MEM_UNIT_MASK) * pool_size,
                                                                              s_mm, r );                                                 
                    }
                    __syncthreads();
                    
                    if( y_idx < dst.y_max && x_idx < dst.x_max ) {
                        float s = s_mm[ threadIdx.y ][ threadIdx.x ];
                        store_method::__store<st_m>( dst[y_idx][x_idx], s );
                    }
                }
        }
        
        template<int st_m, int pool_size>
        __global__ void __sample_maxpooling_ord_kernel_3DGrid( int grid_width, 
                                                               __GT3D dst, 
                                                               const __GT3D prob, 
                                                               const float *rnd ){
            const int block_z = blockIdx.y;
            const int block_y = blockIdx.x / grid_width;
            const int block_x = blockIdx.x % grid_width;
            
            __shared__ float s_mm[ pool_size ][ pool_size*MEM_UNIT ];
            
            __sample_maxpooling_procedure_ord<st_m,pool_size>
                (  block_y, block_x, s_mm, dst[block_z], prob[block_z], 
                   rnd + block_z*(gridDim.x*pool_size*pool_size*MEM_UNIT) + (blockIdx.x*pool_size*pool_size*MEM_UNIT) );        
        }
        
        template<int st_m, int pool_size>
        inline void __sample_maxpooling_ord( GTensor3D &dst, const GTensor3D &prob ){
            dim3 dimBlock( pool_size*MEM_UNIT, pool_size );       

            const int d_y_max = (prob.y_max + pool_size-1) / pool_size;  
            const int d_x_max = (prob.x_max + pool_size-1) / pool_size;

            int  grid_height= (d_y_max+pool_size   -1) / pool_size;        
            int  grid_width = (d_x_max+pool_size*MEM_UNIT-1) / (pool_size*MEM_UNIT);

            dim3 dimGrid( grid_width*grid_height, prob.z_max );
        
            const float *rnd  = cuda_rand::rand_singles( (dimGrid.y*dimGrid.x)*(pool_size*pool_size*MEM_UNIT) );
            
            __sample_maxpooling_ord_kernel_3DGrid<st_m,pool_size><<<dimGrid,dimBlock>>>( grid_width, __GT(dst), __GT(prob), rnd );
        }        
        
        /* pooling data up */
        template<int st_m>
        inline void sample_maxpooling( GTensor3D &dst, const GTensor3D &prob, int pool_size ){        
            switch( pool_size ){
            case 1: sample_binary<st_m>( dst, prob );              break;
            case 2: __sample_maxpooling_rec<st_m,1>( dst, prob ); break;   
            case 3: __sample_maxpooling_ord <st_m,3>( dst, prob ); break;   
            case 4: __sample_maxpooling_rec<st_m,2>( dst, prob ); break;   
            case 8: __sample_maxpooling_rec<st_m,3>( dst, prob ); break;   
            default: error("pooling size not supported");
            }
        }                        
    };
};
#endif

