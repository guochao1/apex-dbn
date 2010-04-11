#ifndef _CUDA_TENSOR_POOLING_CU_
#define _CUDA_TENSOR_POOLING_CU_

#include "cuda_tensor.cuh"
#include "base/cuda_reduce.cuh"

// pooling procedure that summary up information
namespace apex_tensor{
    namespace cuda_tensor{        
        /*----------the following are sum kernels that calculate sum of the data-------*/
        template<int st_m,int mapm_A, int mapm_B>
        __global__ void __tensor_sum_2D_kernel( float *v_reduce_A , float *v_reduce_B, const __GT3D src ){
            __shared__ float s_mmx[16][16];
            const __GT2D ss = src[ blockIdx.x ];
            
            float sumA = 0.0f, sumB = 0.0f;

            for( int yy = 0 ; yy < ss.y_max ; yy += 16 )
                for( int xx = 0 ; xx < ss.x_max ; xx += 16 ){
                    const int y_idx = yy + threadIdx.y; 
                    const int x_idx = xx + threadIdx.x; 

                    if( y_idx < ss.y_max && x_idx < ss.x_max ){
                        float a = ss[y_idx][x_idx];
                        sumA += map_method_A::__map<mapm_A>( a );
                        sumB += map_method_A::__map<mapm_B>( a );                    
                    }
                }
            
            s_mmx[ threadIdx.y ][ threadIdx.x ] = sumA;
            __syncthreads();
            
            cuda_reduce::reduce_2D<cuda_reduce::SUM,4,4>( s_mmx );
            // we only depend on thread 0 0, no need to sync

            if( threadIdx.y == 0 && threadIdx.x == 0 ){
                store_method::__store<st_m>( v_reduce_A[ blockIdx.x ], s_mmx[0][0] ); 
            }
            
            s_mmx[ threadIdx.y ][ threadIdx.x ] = sumB;
            __syncthreads();
            
            cuda_reduce::reduce_2D<cuda_reduce::SUM,4,4>( s_mmx );
            if( threadIdx.y == 0 && threadIdx.x == 0 ){
                store_method::__store<st_m>( v_reduce_B[ blockIdx.x ], s_mmx[0][0] ); 
            }
        }
               
        template<int st_m,int mapm_A, int mapm_B>
        inline void tensor_sum_2D( GTensor1D &ra, GTensor1D &rb, const GTensor3D &src ){
            dim3 dimBlock( 16, 16 );
            dim3 dimGrid ( src.z_max, 1 );
            __tensor_sum_2D_kernel<st_m,mapm_A,mapm_B> <<<dimGrid,dimBlock,0,cuda_async::get_stream(ra,rb,src)>>>
                ( ra.elem, rb.elem, __GT(src) );
        }
    };
    
    // mat sum 
    namespace cuda_tensor{
        // produce sum of matrix, store sum at s_mmx[0][0]
        template<int mapm>
        __device__ void __tensor_sum_2D_procedure( float s_mmx[16][16], const __GT2D ss ){
            float sum = 0.0f;
            for( int yy = 0; yy < ss.y_max; yy += 16 )
                for( int xx = 0; xx < ss.x_max; xx += 16 ){
                    const int y_idx = yy + threadIdx.y; 
                    const int x_idx = xx + threadIdx.x; 
                    
                    if( y_idx < ss.y_max && x_idx < ss.x_max ){
                        sum += map_method_A::__map<mapm>( ss[y_idx][x_idx] );
                    }
                }
            s_mmx[ threadIdx.y ][ threadIdx.x ] = sum;
            __syncthreads();            
            cuda_reduce::reduce_2D<cuda_reduce::SUM,4,4>( s_mmx );
        }
        
    
        template<int st_m, int map_m>
        __global__ void __tensor_sum_2D_kernel( float *v_reduce, const __GT3D src ){
            __shared__ float s_mmx[16][16];
            
            __tensor_sum_2D_procedure<map_m>( s_mmx, src[ blockIdx.x ] );
            // we only depend on thread 0 0
            if( threadIdx.y == 0 && threadIdx.x == 0 ){
                store_method::__store<st_m>( v_reduce[ blockIdx.x ], s_mmx[0][0] ); 
            }
        }
        
        template<int st_m, int map_m>
        inline void tensor_sum_2D( GTensor1D &r , const GTensor3D &src ){
            dim3 dimBlock( 16 , 16 );
            dim3 dimGrid ( src.z_max ,1 );
            __tensor_sum_2D_kernel<st_m,map_m><<<dimGrid,dimBlock,0,cuda_async::get_stream(r,src)>>>( r.elem, __GT(src) );
        }

        template<int st_m, int map_m>
        __global__ void __tensor_sum_2D_kernel( __GT2D dst , const __GT4D src ){
            __shared__ float s_mmx[16][16];
            
            __tensor_sum_2D_procedure<map_m>( s_mmx, src[ blockIdx.y ][ blockIdx.x ] );
            // we only depend on thread 0 0
            if( threadIdx.y == 0 && threadIdx.x == 0 ){
                store_method::__store<st_m>( dst[ blockIdx.y ][ blockIdx.x ], s_mmx[0][0] ); 
            }
        }
        
        template<int st_m, int map_m>
        inline void tensor_sum_2D( GTensor2D &r , const GTensor4D &src ){
            dim3 dimBlock( 16, 16 );
            dim3 dimGrid ( src.z_max, src.h_max );
            __tensor_sum_2D_kernel<st_m,map_m><<<dimGrid,dimBlock,0,cuda_async::get_stream(r,src)>>>( __GT(r), __GT(src) );
        }
    };

    // pool 
    namespace cuda_tensor{
        /* 
           pool src to dst,optimized for pool_size = 2^pool_bits, write the result to sum
           with block shape < 16 , 16 >
        */
        template<int pool_bits>
        __device__ void __pool_procedure_1616( float &sum,
                                               int block_y,
                                               int block_x,    
                                               float s_mm[16][17],
                                               const __GT2D src  ){
            // load from src 
            for( int y = 0 ; y < (1<<pool_bits) ; y ++ )
                for( int x = 0 ; x < (1<<pool_bits) ; x ++ ){                
                    int y_idx = block_y * (16 << pool_bits) + (y<<4) + threadIdx.y; 
                    int x_idx = block_x * (16 << pool_bits) + (x<<4) + threadIdx.x;

                    if( y_idx < src.y_max && x_idx < src.x_max ) {
                        s_mm[ threadIdx.y ][ threadIdx.x ] = src[ y_idx ][ x_idx ];
                    }else{
                        s_mm[ threadIdx.y ][ threadIdx.x ] = 0.0f; 
                    }
                    __syncthreads();
                    // reduce the sum
                    cuda_reduce::reduce_block_1616<cuda_reduce::SUM,pool_bits,pool_bits>( s_mm );
                    __syncthreads();
                    
                    // if the thread is in this range 
                    if( y == ((threadIdx.y<<pool_bits)>>4) && x == ((threadIdx.x<<pool_bits)>>4) ){
                        // no bank conflict in the same pool, since we only access bank in the same row 
                        sum = s_mm[ (threadIdx.y<<pool_bits) & 15 ][ (threadIdx.x<<pool_bits) & 15 ]; 
                    }
                    // must sync here !!
                    __syncthreads();
                }
        }
        
        /* pooling kernel, using 3DGrid */
        template<int st_m, int map_m, int pool_bits>
        __global__ void __pool_kernel_1616( int grid_width, __GT3D dst, const __GT3D src ){
            const int block_z = blockIdx.y;
            const int block_y = blockIdx.x / grid_width;
            const int block_x = blockIdx.x % grid_width;

            __shared__ float s_mm[ 16 ][ 17 ];
            
            // pool procedure
            float sum = 0.0f;
            __pool_procedure_1616<pool_bits>( sum, block_y, block_x, s_mm, src[block_z] );        

            // store result back 
            const int yy_idx = (block_y << 4) + threadIdx.y;
            const int xx_idx = (block_x << 4) + threadIdx.x;
            
            if( yy_idx < dst.y_max && xx_idx < dst.x_max  ){  
                float val =  map_method_A::__map<map_m>( sum );
                store_method::__store<st_m>( dst[ block_z ][ yy_idx ][ xx_idx ], val );
            }        
        }

        /* pooling data up */
        template<int st_m,int map_m,int pool_bits>
        inline void __pool_up_1616( GTensor3D &dst, const GTensor3D &src ){        
            int  grid_height= (dst.y_max+15) >> 4 ;
            int  grid_width = (dst.x_max+15) >> 4;

            dim3 dimBlock( 16 , 16 );
            dim3 dimGrid( grid_width*grid_height, src.z_max );           
            __pool_kernel_1616<st_m,map_m,pool_bits><<<dimGrid,dimBlock>>>( grid_width, __GT(dst), __GT(src) );
        }

        /* ordinary pooling */
        template<int pool_size>
        inline __device__ float __sum_block_ord( int y_start, int x_start, float s_mm[pool_size][pool_size*16] ){
            float sum = 0.0f;
            for( int y = y_start ; y < y_start + pool_size ; y ++ )
                for( int x = x_start ; x < x_start + pool_size*16 ; x ++ ){
                    sum += s_mm[y][x];
                }
            return sum;
        }

        /* 
           pool src to dst,optimized for pool_size 
           with block shape < pool_size , pool_size*16 >
        */
        template<int pool_size>
        __device__ void __pool_procedure_ord( float &sum,
                                              int block_y,
                                              int block_x,    
                                              float s_mm[pool_size][pool_size*16],
                                              const __GT2D src  ){
            for( int y = 0 ; y < pool_size ; y ++ )
                for( int x = 0 ; x < pool_size; x ++ ){                
                    int y_idx = block_y * pool_size*pool_size    + y*pool_size    + threadIdx.y; 
                    int x_idx = block_x * pool_size*pool_size*16 + x*pool_size*16 + threadIdx.x;

                    if( y_idx < src.y_max && x_idx < src.x_max ) {
                        s_mm[ threadIdx.y ][ threadIdx.x ] = src[ y_idx ][ x_idx ];
                    }else{
                        s_mm[ threadIdx.y ][ threadIdx.x ] = 0.0f; 
                    }

                    __syncthreads();

                    // if the thread is in this range 
                    if( y == threadIdx.y && x == (threadIdx.x>>4) ){
                        sum = __sum_block_ord<pool_size>( 0, ( threadIdx.x & 15 ) * pool_size , s_mm );
                    }
                    // must sync here !!
                    __syncthreads();
                }
        }
        
        /* pooling kernel, using 3DGrid */
        template<int st_m, int map_m, int pool_size>
        __global__ void __pool_kernel_ord( int grid_width, __GT3D dst, const __GT3D src ){
            const int block_z = blockIdx.y;
            const int block_y = blockIdx.x / grid_width;
            const int block_x = blockIdx.x % grid_width;

            __shared__ float s_mm[ pool_size ][ pool_size*16 ];
            
            // pool procedure
            float sum = 0.0f;
            __pool_procedure_ord<pool_size>( sum, block_y, block_x, s_mm, src[block_z] );        

            // store result back 
            const int yy_idx = block_y*pool_size    + threadIdx.y;
            const int xx_idx = block_x*pool_size*16 + threadIdx.x;
            
            if( yy_idx < dst.y_max && xx_idx < dst.x_max  ){  
                float val =  map_method_A::__map<map_m>( sum );
                store_method::__store<st_m>( dst[ block_z ][ yy_idx ][ xx_idx ], val );
            }        
        }
        
        
        template<int st_m,int map_m,int pool_size>
        inline void __pool_up_ord( GTensor3D &dst, const GTensor3D &src ){        
            int  grid_height= (dst.y_max+pool_size-1   ) / pool_size ;
            int  grid_width = (dst.x_max+pool_size*16-1) / (pool_size*16);

            dim3 dimBlock( pool_size*16 , pool_size );
            dim3 dimGrid( grid_width*grid_height, src.z_max );           
            __pool_kernel_ord<st_m,map_m,pool_size><<<dimGrid,dimBlock>>>( grid_width, __GT(dst), __GT(src) ); 
        }

        template<int st_m,int map_m>
        inline void pool_up( GTensor3D &dst, const GTensor3D &src, int pool_size ){
            switch( pool_size ){
            case 1 : map_A<st_m,map_m,GTensor3D> ( dst, src ); break;
            case 2 : __pool_up_1616<st_m,map_m,1>( dst, src ); break;
            case 3 : __pool_up_ord <st_m,map_m,3>( dst, src ); break;
            case 4 : __pool_up_1616<st_m,map_m,2>( dst, src ); break;
            case 5 : __pool_up_ord <st_m,map_m,5>( dst, src ); break;
            case 8 : __pool_up_1616<st_m,map_m,3>( dst, src ); break;                
            default: error("pool_size not supported"); 
            }
        }
    };

    // pooling sum 
    namespace cuda_tensor{
        /* 
           pool src, sum up the pooled value, then use two kind of maps to get mapped value, 
           store result to sumA, and sumB 
           ,optimized for pool_size = 2^pool_bits
           with block shape < 16 , 16 >       
        */
        template<int mapm_A, int mapm_B, int pool_bits, bool ceil_up>
        __device__ void __pool_sum_procedure_1616( float &sumA,         
                                                   float &sumB,
                                                   float s_mm[16][17],
                                                   const __GT2D src  ){
            int d_y_max, d_x_max;
            if( ceil_up ){
                d_y_max = (src.y_max + (1<<pool_bits)-1) >> pool_bits;
                d_x_max = (src.x_max + (1<<pool_bits)-1) >> pool_bits;

            }else{
                d_y_max = src.y_max >> pool_bits;
                d_x_max = src.x_max >> pool_bits;
            }
            
            for( int yy = 0 ; yy < d_y_max ; yy +=16 )
                for( int xx = 0 ; xx < d_x_max ; xx +=16 ){
                    float s = 0.0f;
                    // pool result 
                    __pool_procedure_1616<pool_bits>( s, yy>>4, xx>>4, s_mm, src );

                    const int yy_idx = yy + threadIdx.y;        
                    const int xx_idx = xx + threadIdx.x;

                    // add product 
                    if( yy_idx < d_y_max && xx_idx < d_x_max ) {
                        sumA += map_method_A::__map<mapm_A>( s ); 
                        sumB += map_method_A::__map<mapm_B>( s ); 
                    }
                } 
        }
        
        /* pooling kernel, using 3DGrid */
        template<int st_m, int mapm_A, int mapm_B, int pool_bits,bool ceil_up>
        __global__ void __pool_sum_kernel_1616( float *v_reduce_A, float *v_reduce_B, const __GT3D src ){
            __shared__ float s_mm[ 16 ][ 17 ];
                        
            float sumA = 0.0f;
            float sumB = 0.0f;  
            __pool_sum_procedure_1616<mapm_A,mapm_B,pool_bits, ceil_up>
                ( sumA, sumB, s_mm, src[blockIdx.x] );        
            
            float (*s_mmx)[16] = (float(*)[16])s_mm[0];
            
            s_mmx[ threadIdx.y ][ threadIdx.x ] = sumA;
            __syncthreads();
            
            cuda_reduce::reduce_2D<cuda_reduce::SUM,4,4>( s_mmx );
            // we only depend on thread 0 0
            if( threadIdx.y == 0 && threadIdx.x == 0 ){
                store_method::__store<st_m>( v_reduce_A[ blockIdx.x ] , s_mmx[0][0] ); 
            }
            
            s_mmx[ threadIdx.y ][ threadIdx.x ] = sumB;
            __syncthreads();
            
            cuda_reduce::reduce_2D<cuda_reduce::SUM,4,4>( s_mmx );
            if( threadIdx.y == 0 && threadIdx.x == 0 ){
                store_method::__store<st_m>( v_reduce_B[ blockIdx.x ] , s_mmx[0][0] ); 
            }
        }
        
        template<int st_m, int mapm_A, int mapm_B, int pool_bits, bool ceil_up>
        inline void __pool_sum_1616( GTensor1D &ra, GTensor1D &rb, const GTensor3D &src ){        
            dim3 dimBlock( 16 , 16 );
            dim3 dimGrid ( src.z_max ,1 );

            cudaStream_t s = cuda_async::get_stream( ra, rb, src );
            __pool_sum_kernel_1616<st_m,mapm_A,mapm_B,pool_bits,ceil_up><<<dimGrid,dimBlock,0,s>>>( ra.elem, rb.elem, __GT(src) );
        }

        template<int mapm_A, int mapm_B, int pool_size, bool ceil_up>
        __device__ void __pool_sum_procedure_ord( float &sumA,         
                                                  float &sumB,
                                                  float s_mm[pool_size][pool_size*16],
                                                  const __GT2D src  ){
            int d_y_max, d_x_max;
            if( ceil_up ){
                d_y_max = (src.y_max + pool_size-1) / pool_size;
                d_x_max = (src.x_max + pool_size-1) / pool_size;

            }else{
                d_y_max = src.y_max / pool_size;
                d_x_max = src.x_max / pool_size;
            }
            
            for( int yy = 0 ; yy < d_y_max ; yy += pool_size )
                for( int xx = 0 ; xx < d_x_max ; xx += pool_size*16 ){
                    float s = 0.0f;
                    __pool_procedure_ord<pool_size>( s, yy/pool_size, xx/(pool_size*16), s_mm, src );

                    const int yy_idx = yy + threadIdx.y;        
                    const int xx_idx = xx + threadIdx.x;

                    // add product 
                    if( yy_idx < d_y_max && xx_idx < d_x_max ) {
                        sumA += map_method_A::__map<mapm_A>( s ); 
                        sumB += map_method_A::__map<mapm_B>( s ); 
                    }
                } 
        }
        
        template<int st_m, int mapm_A, int mapm_B, int pool_size, bool ceil_up>
        __global__ void __pool_sum_kernel_ord( float *v_reduce_A, float *v_reduce_B, const __GT3D src ){
            __shared__ float s_mm[ pool_size ][ pool_size*16 ];
                        
            float sumA = 0.0f;
            float sumB = 0.0f;  
            __pool_sum_procedure_ord<mapm_A,mapm_B,pool_size, ceil_up>
                ( sumA, sumB, s_mm, src[blockIdx.x] );        
                               
            s_mm[ threadIdx.y ][ threadIdx.x ] = sumA;
            __syncthreads();
            
            cuda_reduce::reduce_2D_non_align<cuda_reduce::SUM,pool_size,pool_size*16>( s_mm );
            // we only depend on thread 0 0
            if( threadIdx.y == 0 && threadIdx.x == 0 ){
                store_method::__store<st_m>( v_reduce_A[ blockIdx.x ] , s_mm[0][0] ); 
            }
            
            s_mm[ threadIdx.y ][ threadIdx.x ] = sumB;
            __syncthreads();
            
            cuda_reduce::reduce_2D_non_align<cuda_reduce::SUM,pool_size,pool_size*16>( s_mm );

            if( threadIdx.y == 0 && threadIdx.x == 0 ){
                store_method::__store<st_m>( v_reduce_B[ blockIdx.x ] , s_mm[0][0] ); 
            }
        }        

        template<int st_m, int mapm_A, int mapm_B, int pool_size, bool ceil_up>
        inline void __pool_sum_ord( GTensor1D &ra, GTensor1D &rb, const GTensor3D &src ){        
            dim3 dimBlock( pool_size*16, pool_size );
            dim3 dimGrid ( src.z_max ,1 );

            cudaStream_t s = cuda_async::get_stream( ra, rb, src );
            __pool_sum_kernel_ord<st_m,mapm_A,mapm_B,pool_size,ceil_up><<<dimGrid,dimBlock,pool_size,s>>>( ra.elem, rb.elem, __GT(src) ); 
        }
        
        template<int st_m, int mapm_A, int mapm_B, bool ceil_up>
        inline void pool_sum( GTensor1D &ra, GTensor1D &rb, const GTensor3D &src, int pool_size ){        
            switch( pool_size ){
            case 1: tensor_sum_2D  <st_m,mapm_A,mapm_B>( ra, rb, src ); break;
            case 2: __pool_sum_1616<st_m,mapm_A,mapm_B,1,ceil_up>( ra, rb, src ); break;
            case 3: __pool_sum_ord <st_m,mapm_A,mapm_B,3,ceil_up>( ra, rb, src ); break;
            case 4: __pool_sum_1616<st_m,mapm_A,mapm_B,2,ceil_up>( ra, rb, src ); break;
            case 5: __pool_sum_ord <st_m,mapm_A,mapm_B,5,ceil_up>( ra, rb, src ); break;
            case 8: __pool_sum_1616<st_m,mapm_A,mapm_B,3,ceil_up>( ra, rb, src ); break;
            default: error("pooling size not supported");
            }
        }
    };        

    // normalize by maxpooling softmax
    namespace cuda_tensor{                
        // normalize the data start from y_start,x_start by exp
        // return the normalization constant for 1
        template<int pool_size, int y_size, int x_size>
        inline __device__ float __norm_maxpooling_step1( int y_start, int x_start, float s_mm[y_size][x_size] ){
            // get the max value of the data
            float smax = s_mm[y_start][x_start];
            for( int y = y_start ; y < y_start + pool_size ; y ++ )
                for( int x = x_start ; x < x_start + pool_size ; x ++ ){
                    if( smax < s_mm[y][x] ) smax = s_mm[y][x];
                }
            // map to exp
            for( int y = y_start ; y < y_start + pool_size ; y ++ )
                for( int x = x_start ; x < x_start + pool_size ; x ++ ){
                    s_mm[ y ][ x ] = expf( s_mm[ y ][ x ] - smax ); 
                }
            return expf( - smax );
        }

        template<int pool_size, int y_size, int x_size>
        inline __device__ void __norm_maxpooling_step2( int y_start, int x_start, float s_mm[y_size][x_size], float nm ){
            // get the max value of the data
            float sum = nm;
            for( int y = y_start ; y < y_start + pool_size ; y ++ )
                for( int x = x_start ; x < x_start + pool_size ; x ++ ){
                    sum += s_mm[ y ][ x ];
                }
            // map to exp
            for( int y = y_start ; y < y_start + pool_size ; y ++ )
                for( int x = x_start ; x < x_start + pool_size ; x ++ ){
                    s_mm[ y ][ x ] /= sum;
                }
        }

        /* 
           normalize the data by maxpooling with pool_size = 2^pool_bits
           with block shape < 16 , 16 >
        */
        template<int st_m,int pool_bits>
        __device__ void __norm_maxpooling_procedure_1616( int block_y,
                                                          int block_x,    
                                                          float s_mm[16][16],
                                                          __GT2D dst,
                                                          const __GT2D energy ){
            // load from src 
            for( int y = 0 ; y < (1<<pool_bits) ; y ++ )
                for( int x = 0 ; x < (1<<pool_bits) ; x ++ ){                                
                    int y_idx = block_y * (16 << pool_bits) + (y<<4) + threadIdx.y;
                    int x_idx = block_x * (16 << pool_bits) + (x<<4) + threadIdx.x;                    
                    bool is_valid   = y_idx < energy.y_max && x_idx < energy.x_max;
                    bool is_inrange = (y == ((threadIdx.y<<pool_bits)>>4) && x == ((threadIdx.x<<pool_bits)>>4) );
                    
                    // we don't need to sync here since each thread always use the same position                     
                    if( is_valid ){
                        s_mm[ threadIdx.y ][ threadIdx.x ] = energy[ y_idx ][ x_idx ];
                    }else{
                        s_mm[ threadIdx.y ][ threadIdx.x ] = -1e20f; 
                    }
                    __syncthreads();
                    
                    float nm;
                    // if the thread is in this range 
                    if( is_inrange ){
                        // no bank conflict in the same pool, since we only access bank in the same row 
                        nm = __norm_maxpooling_step1<1<<pool_bits,16,16>( (threadIdx.y<<pool_bits) &15, 
                                                                          (threadIdx.x<<pool_bits) &15,
                                                                          s_mm );                                                 
                    }
                    __syncthreads();

                    if( !is_valid ) {
                        s_mm[ threadIdx.y ][ threadIdx.x ] = 0.0f;
                    }
                    __syncthreads();
                    
                    if( is_inrange ){
                        // no bank conflict in the same pool, since we only access bank in the same row 
                        __norm_maxpooling_step2<1<<pool_bits,16,16>( (threadIdx.y<<pool_bits) &15, 
                                                                     (threadIdx.x<<pool_bits) &15,
                                                                     s_mm, nm );                                                 
                    }
                    __syncthreads();
                    
                    if( is_valid ) {
                        float s = s_mm[ threadIdx.y ][ threadIdx.x ];
                        store_method::__store<st_m>( dst[y_idx][x_idx], s );
                    }
                    // no need to sync
                }
        }
        
        /* pooling kernel, using 3DGrid */
        template<int st_m, int pool_bits>
        __global__ void __norm_maxpooling_kernel_1616( int grid_width, 
                                                       __GT3D dst, 
                                                       const __GT3D energy ){
            const int block_z = blockIdx.y;
            const int block_y = blockIdx.x / grid_width;
            const int block_x = blockIdx.x % grid_width;
            
            __shared__ float s_mm[ 16 ][ 16 ];
            
            __norm_maxpooling_procedure_1616<st_m,pool_bits>
                (  block_y, block_x, s_mm, dst[block_z], energy[block_z] );
        }
        
        /* pooling data up */
        template<int st_m, int pool_bits>
        inline void __norm_maxpooling_1616( GTensor3D &dst, const GTensor3D &energy ){                    
            dim3 dimBlock( 16 , 16 );       
            const int d_y_max = (energy.y_max + (1<<pool_bits)-1) >> pool_bits;  
            const int d_x_max = (energy.x_max + (1<<pool_bits)-1) >> pool_bits;

            int  grid_height= (d_y_max+15) >> 4 ;        
            int  grid_width = (d_x_max+15) >> 4;

            dim3 dimGrid( grid_width*grid_height, energy.z_max );
            
            __norm_maxpooling_kernel_1616<st_m,pool_bits><<<dimGrid,dimBlock>>>( grid_width, __GT(dst), __GT(energy) );
        }

        /* 
           normalize the data by maxpooling with pool_size
           with block shape < pool_size , 16*pool_size >
        */
        template<int st_m,int pool_size>
        __device__ void __norm_maxpooling_procedure_ord( int block_y,
                                                         int block_x,    
                                                         float s_mm[pool_size][pool_size*16],
                                                         __GT2D dst,
                                                         const __GT2D energy ){
            // load from src 
            for( int y = 0 ; y < pool_size ; y ++ )
                for( int x = 0 ; x < pool_size ; x ++ ){                                
                    int y_idx = block_y * pool_size*pool_size    + y*pool_size    + threadIdx.y;
                    int x_idx = block_x * pool_size*pool_size*16 + x*pool_size*16 + threadIdx.x;                    
                    bool is_valid   = y_idx < energy.y_max && x_idx < energy.x_max;
                    bool is_inrange = ( y==threadIdx.y && x == (threadIdx.x>>4) );
                    
                    // we don't need to sync here since each thread always use the same position                     
                    if( is_valid ){
                        s_mm[ threadIdx.y ][ threadIdx.x ] = energy[ y_idx ][ x_idx ];
                    }else{
                        s_mm[ threadIdx.y ][ threadIdx.x ] = -1e20f; 
                    }
                    __syncthreads();
                    
                    float nm;
                    // if the thread is in this range 
                    if( is_inrange ){
                        nm = __norm_maxpooling_step1<pool_size,pool_size,pool_size*16>
                            ( 0, (threadIdx.x&15)*pool_size, s_mm );                                                 
                    }
                    __syncthreads();

                    if( !is_valid ) {
                        s_mm[ threadIdx.y ][ threadIdx.x ] = 0.0f;
                    }
                    __syncthreads();
                    
                    if( is_inrange ){
                        __norm_maxpooling_step2<pool_size,pool_size,pool_size*16>
                            ( 0, (threadIdx.x&15)*pool_size, s_mm, nm );                                                 
                    }
                    __syncthreads();
                    
                    if( is_valid ) {
                        float s = s_mm[ threadIdx.y ][ threadIdx.x ];
                        store_method::__store<st_m>( dst[y_idx][x_idx], s );
                    }
                }
        }

        template<int st_m, int pool_size>
        __global__ void __norm_maxpooling_kernel_ord( int grid_width, 
                                                      __GT3D dst, 
                                                      const __GT3D energy ){
            const int block_z = blockIdx.y;
            const int block_y = blockIdx.x / grid_width;
            const int block_x = blockIdx.x % grid_width;
            
            __shared__ float s_mm[ pool_size ][ pool_size*16 ];
            
            __norm_maxpooling_procedure_ord<st_m,pool_size>
                (  block_y, block_x, s_mm, dst[block_z], energy[block_z] );
        }
        
        template<int st_m, int pool_size>
        inline void __norm_maxpooling_ord( GTensor3D &dst, const GTensor3D &energy ){                    
            dim3 dimBlock( pool_size*16, pool_size );       
            const int d_y_max = (energy.y_max + pool_size-1) / pool_size;  
            const int d_x_max = (energy.x_max + pool_size-1) / pool_size;

            int  grid_height= (d_y_max+ pool_size-1)    / pool_size;        
            int  grid_width = (d_x_max+ pool_size*16-1) / (pool_size*16);

            dim3 dimGrid( grid_width*grid_height, energy.z_max );
            
            __norm_maxpooling_kernel_ord<st_m,pool_size><<<dimGrid,dimBlock>>>( grid_width, __GT(dst), __GT(energy) );
        }
        
        template<int st_m>
        inline void norm_maxpooling( GTensor3D &dst, const GTensor3D &energy, int pool_size ){        
            switch( pool_size ){
            case 1: map_A<st_m,map_method_A::SIGMOID>( dst, energy ); break;
            case 2: __norm_maxpooling_1616<st_m,1>( dst, energy );    break;
            case 3: __norm_maxpooling_ord <st_m,3>( dst, energy );    break;
            case 4: __norm_maxpooling_1616<st_m,2>( dst, energy );    break;
            case 5: __norm_maxpooling_ord <st_m,5>( dst, energy );    break;
            case 8: __norm_maxpooling_1616<st_m,3>( dst, energy );    break;
            default: error("pooling size not supported");
            }
        }        
    };
};
#endif

