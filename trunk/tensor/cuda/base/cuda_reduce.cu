#ifndef _CUDA_REDUCE_CU_
#define _CUDA_REDUCE_CU_

namespace cuda_reduce{
    namespace reduce_method{
        // reduce 
        template<int rm>
        __device__ void __reduce( float &dst, float src );

        template<>
        __device__ void __reduce<SUM>( float &dst, float src ){
            dst += src;
        } 
        
        template<>
        __device__ void __reduce<MAX>( float &dst, float src ){
            dst = max( dst, src );
        } 

    };

    /* 
       buf_idx is the id of thread indicate the position in buffer, 
       pos_idx is the id of thread indicate the position in block
       x_bits corresponds to bits each block
    */
    template<int rm,int x_bits>
    __device__ void __reduce_x( float buf[],int tid ){
        if( x_bits >= 9 ){
            if( tid < 256 ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 256] );
            __syncthreads(); 
        }
        if( x_bits >= 8 ){
            if( tid < 128 ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 128] );
            __syncthreads(); 
        }
        if( x_bits >= 7 ){
            if( tid < 64  ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 64 ] );
            __syncthreads(); 
        }
#ifndef  __DEVICE_EMULATION__
        /* code for in warp optimization */
        if( tid < 32 ){
            if( x_bits >= 6 ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 32 ] );
            if( x_bits >= 5 ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 16 ] );
            if( x_bits >= 4 ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 8 ]  );
            if( x_bits >= 3  ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 4 ]  );
            if( x_bits >= 2  ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 2 ]  );
            if( x_bits >= 1  ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 1 ]  );
        }
#else
        if( x_bits >= 6 ){
            if( tid < 32 ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 32] );
            __syncthreads(); 
        }
        if( x_bits >= 5 ){
            if( tid < 16 ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 16] );
            __syncthreads(); 
        }
        if( x_bits >= 4 ){
            if( tid < 8 ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 8 ] );
            __syncthreads(); 
        }
        if( x_bits >= 3 ){
            if( tid < 4 ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 4 ] );
            __syncthreads(); 
        }
        if( x_bits >= 2 ){
            if( tid < 2 ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 2 ] );
            __syncthreads(); 
        }
        if( x_bits >= 1 ){
            if( tid < 1 ) reduce_method::__reduce<rm>( buf[tid] , buf[tid + 1 ] );
        }        
#endif        
    }
   
#define __RD_NON_ALIGN(els,x_bits)                                      \
    els                                                                 \
    if( x_size >= (1 << x_bits) ){                                      \
        if( tid < (1 << x_bits) && tid + (1<<x_bits) < x_size ){        \
            reduce_method::__reduce<rm>( buf[tid] , buf[tid + (1<<x_bits)] ); \
        }                                                               \
        __syncthreads();                                                \
        __reduce_x<rm, x_bits>( buf, tid );                             \
    }                                                                   \


    template<int rm,int x_size>
    __device__ void __reduce_x_non_align( float buf[], int tid ){
        __RD_NON_ALIGN(, 8) 
        __RD_NON_ALIGN(else, 7) 
        __RD_NON_ALIGN(else, 6) 
        __RD_NON_ALIGN(else, 5) 
        __RD_NON_ALIGN(else, 4) 
        __RD_NON_ALIGN(else, 3) 
        __RD_NON_ALIGN(else, 2) 
        __RD_NON_ALIGN(else, 1)                     
    }

    template<int rm,int x_bits>
    __device__ void reduce_1D( float buf[1<<x_bits] ){
        __reduce_x< rm, x_bits >( buf , threadIdx.x );
    }    


    template<int rm,int y_bits, int x_bits>
    __device__ void reduce_2D( float buf[1<<y_bits][1<<x_bits] ){
        __reduce_x< rm , x_bits+y_bits > ( buf[0] , (threadIdx.y << x_bits) + threadIdx.x );
    }   

    template<int rm,int y_size, int x_size>
    __device__ void reduce_2D_non_align( float buf[y_size][x_size] ){
        __reduce_x_non_align< rm , x_size*y_size > ( buf[0] , threadIdx.y*x_size + threadIdx.x );
    }   

    /* block reduction optimized for <16,16> thread block */
    template<int rm, int block_bits>
    __device__ void __reduce_x_block_1616( float buf[ 16 ][ 17 ] ){
        // position in block 
        // we reverse x and y so that there is no warp divergence
        const int pos   = threadIdx.y &( (1<<block_bits) - 1 );
        const int x_idx = threadIdx.y;
        const int y_idx = threadIdx.x;

        if( block_bits >= 4 ){
            if( pos < 8 ) reduce_method::__reduce<rm>( buf[y_idx][x_idx] ,  buf[y_idx][x_idx + 8] ); 
            __syncthreads();
        }
       
        if( block_bits >= 3 ){
            if( pos < 4 ) reduce_method::__reduce<rm>( buf[y_idx][x_idx] ,  buf[y_idx][x_idx + 4] ); 
            __syncthreads();
        } 
        if( block_bits >= 2 ){
            if( pos < 2 ) reduce_method::__reduce<rm>( buf[y_idx][x_idx] ,  buf[y_idx][x_idx + 2] ); 
            __syncthreads();
        }                                                                          
                                                                                  
        if( block_bits >= 1 ){
            if( pos < 1 ) reduce_method::__reduce<rm>( buf[y_idx][x_idx] ,  buf[y_idx][x_idx + 1] );         
        }
    }

    template<int rm,int block_bits>
    __device__ void __reduce_y_block_1616( float buf[16][17] ){
        // stride = 17 is used to avoid bank conflict
        const int pos   = threadIdx.y &( (1<<block_bits) - 1 );
        const int x_idx = threadIdx.x;
        const int y_idx = threadIdx.y;

        if( block_bits >= 4 ){
            if( pos < 8 ) reduce_method::__reduce<rm>( buf[y_idx][x_idx] ,  buf[y_idx+8][x_idx] ); 
            __syncthreads();
        }
       
        if( block_bits >= 3 ){
            if( pos < 4 ) reduce_method::__reduce<rm>( buf[y_idx][x_idx] ,  buf[y_idx+4][x_idx] ); 
            __syncthreads();
        } 
        if( block_bits >= 2 ){
            if( pos < 2 ) reduce_method::__reduce<rm>( buf[y_idx][x_idx] ,  buf[y_idx+2][x_idx] ); 
            __syncthreads();
        }                                                                          
                                                                                  
        if( block_bits >= 1 ){
            if( pos < 1 ) reduce_method::__reduce<rm>( buf[y_idx][x_idx] ,  buf[y_idx+1][x_idx] );         
        }
    }

    /* back propagate the reduced answer to all elements in same line */
    template<int block_bits>
    __device__ void backprop_x_block_1616(  float buf[ 16 ][ 17 ] ){
        // position in block 
        // we reverse x and y so that there is no warp divergence
        const int pos   = threadIdx.y &( (1<<block_bits) - 1 );
        const int x_idx = threadIdx.y;
        const int y_idx = threadIdx.x;
                                                                                  
        if( block_bits >= 1 ){
            if( pos < 1 ) buf[y_idx][x_idx + 1] = buf[y_idx][x_idx];  
            __syncthreads();
        }

        if( block_bits >= 2 ){
            if( pos < 2 ) buf[y_idx][x_idx + 2] = buf[y_idx][x_idx]; 
            __syncthreads();
        }                                                                          

        if( block_bits >= 3 ){
            if( pos < 4 ) buf[y_idx][x_idx + 4] = buf[y_idx][x_idx]; 
            __syncthreads();
        } 

        if( block_bits >= 4 ){
            if( pos < 8 ) buf[y_idx][x_idx + 8] =  buf[y_idx][x_idx]; 
            __syncthreads();
        }
    }
    
    template<int rm, int y_block_bits,int x_block_bits>
    __device__ void reduce_block_1616(  float buf[ 16 ][ 17 ] ){
        __reduce_y_block_1616<rm,y_block_bits>( buf ); 

        __syncthreads();
        
        __reduce_x_block_1616<rm,x_block_bits>( buf ); 
        
    }
    
    template<int rm, int y_block_bits,int x_block_bits>
    __device__ void reduce_block_with_bp_1616(  float buf[16][17] ){
        __reduce_y_block_1616<rm,y_block_bits>( buf ); 

        __syncthreads();

        __reduce_x_block_1616<rm,x_block_bits>( buf );
        
        __syncthreads();

        backprop_x_block_1616<x_block_bits>( buf );         
    }
};

#endif


