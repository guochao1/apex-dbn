#ifndef _CUDA_RAND_MTGP32_CU_
#define _CUDA_RAND_MTGP32_CU_

#include <cstdio>
#include <cstdlib>
#include "cuda_rand.cuh"
#include "mtgp/mtgp32-interface.cu"

#ifdef __DEBUG_CUDA_RAND__   
__constant__ float * __cuda_rand_d_rand_buf;
__constant__ int   * __cuda_rand_ref_counter;
#endif

namespace cuda_rand{
    // we can add number of block to 128 to get unit_size = 512*3*128=   3 << 15
    // access unit of random number 
    const int __ALIGN_BITS= 4; 
    const int __BUF_SIZE  = 3 << 21;
    const int __UNIT_SIZE = mtgp::LARGE_SIZE * mtgp::BLOCK_NUM ;
    const int __BUF_LEN   = (__BUF_SIZE / __UNIT_SIZE) * __UNIT_SIZE;
    
    mtgp::mtgp32_kernel_status_t* __d_status;
    // counting number of init called 
    static int   __init_counter = 0;
    // pointer to unused random numbers  
    static unsigned int __rand_counter = __BUF_LEN;
    // buffer of random numbers 
    static float *__d_rand_buf;

#ifdef __DEBUG_CUDA_RAND__   
    // reference counte of random used for debug  
    static int  *__ref_counter;
    static int  __host_ref_counter[ __BUF_SIZE ];
    
    inline int rand_debug_check( int r_max = __BUF_LEN ){
        cudaThreadSynchronize();
        cudaMemcpy( __host_ref_counter, __ref_counter, sizeof(int)*__BUF_SIZE , cudaMemcpyDeviceToHost );
        
        int count = 0;
        for( int i = 0 ; i < __BUF_SIZE ; i ++ ){
            if( __host_ref_counter[i] > 0 ){
                count ++;
                if( __host_ref_counter[i] > 1 ){
                    printf("error cuda_rand:: random number[%d]=%d referenced more than once\n", i , __host_ref_counter[i] ); 
                    exit( -1 );
                }
                if( i >= r_max ){
                    printf("error cuda_rand:: random number invalid [%d] > %d\n" , i, r_max ); exit( -1 );
                } 
            }else{
                if( i < r_max && r_max < 1000 ){ 
                    printf ("%d skipped=%d\n", i, __host_ref_counter[i] );
                }                
            }
        }

        return count;
    }
    bool first = true;
    inline void rand_debug_refresh(){
        if( !first ){
            int count = rand_debug_check();
            printf("%d, %lf rand reference\n",count, (double)count / __BUF_LEN );
        }else{
            first = false;
        }

        memset( __host_ref_counter , 0, sizeof( __host_ref_counter ));
        cudaMemcpy( __ref_counter, __host_ref_counter, sizeof(int)*__BUF_SIZE , cudaMemcpyHostToDevice );            
    }

    inline void rand_debug_init(){
        cudaMalloc( (void**)&__ref_counter, sizeof(int)*__BUF_SIZE );
        rand_debug_refresh();
        cudaMemcpyToSymbol( "__cuda_rand_d_rand_buf"  , &__d_rand_buf  , sizeof(float*) );
        cudaMemcpyToSymbol( "__cuda_rand_ref_counter" , &__ref_counter , sizeof(int*) );        
    }

    inline void rand_debug_destroy(){
        cudaFree( __ref_counter );
    }

    inline __device__ void rand_debug_get_rand( const float *rnd, int idx ){
        int *ref = __cuda_rand_ref_counter + (rnd-__cuda_rand_d_rand_buf) + idx; 
        atomicAdd( ref, 1 );
    }

#endif

    inline void rand_init(){
        // note this function is not thread safe 
        if( __init_counter++ > 0 ) return;

        cudaMalloc((void**)&__d_status,
			sizeof(mtgp::mtgp32_kernel_status_t) * mtgp::BLOCK_NUM);
        
        cudaMalloc((void**)&__d_rand_buf, sizeof(float) * __BUF_SIZE);


        mtgp::make_constant(mtgp32_params_fast_23209);
        mtgp::make_kernel_data(__d_status, mtgp32_params_fast_23209);               
        
#ifdef __DEBUG_CUDA_RAND__   
        rand_debug_init();
#endif        
    }
    

    inline void rand_destroy(){
        // this thread is not thread safe 
        if( -- __init_counter == 0 ){
            cudaFree( __d_status );
            cudaFree( __d_rand_buf );            
        }
#ifdef __DEBUG_CUDA_RAND__   
        rand_debug_destroy();
#endif        
    }

    // regenerate the random numbers
    inline void __refresh_buffer(){
#ifdef __DEBUG_CUDA_RAND__   
        rand_debug_check();
        rand_debug_refresh();
#endif
        
        mtgp:: mtgp32_single_kernel<<< mtgp::BLOCK_NUM, mtgp::THREAD_NUM >>>
			( __d_status, (uint32_t*)__d_rand_buf , __BUF_LEN / mtgp::BLOCK_NUM );
		__rand_counter = 0;	
	}
    
    inline const float *rand_singles( unsigned int num ){
		// align to memory unit 
        num = ( (num + (1<<__ALIGN_BITS)-1) >> __ALIGN_BITS) << __ALIGN_BITS; 
        
        if( num >= __BUF_LEN /2 ){
            printf("too many random number requested, try to add buffer size and recompile the code\n");
            exit( -1 );
        }	
		
        if( __rand_counter + num > (unsigned int)__BUF_LEN ){
            __refresh_buffer();
        }
        __rand_counter += num;

        return __d_rand_buf + __rand_counter - num;
    }  

    inline __device__ float get_rand( const float *rnd, int idx ){
#ifdef __DEBUG_CUDA_RAND__   
        rand_debug_get_rand( rnd, idx );
#endif
        return rnd[ idx ];
    }
};
#endif




