#ifndef _CUDA_REDUCE_CUH_
#define _CUDA_REDUCE_CUH_

namespace cuda_reduce{
    /* reduce method supported */
    const int SUM = 0;
    const int MAX = 1;
      
    /*------------------------------*/
    /* 
       reduce over the dimension x 
       dim_x have to be power of 2
    */
    template<int reduce_m,int x_bits>
    __device__ void reduce_1D( float buf[1<<x_bits] );

    /* 
       reduce over 2 dimensions 
       the size of matrix is 1 << y_bits, 1 << x_bits
     */
    template<int reduce_m,int y_bits, int x_bits>
    __device__ void reduce_2D( float buf[1<<y_bits][1<<x_bits] );
    
    /* block reduce function optimized for <16,16> thread block */
    template<int reduce_m,int y_block_bits,int x_block_bits>
    __device__ void reduce_block_1616( float buf[16][17] );

    /* 
       block reduce function optimized for <16,16> thread block 
       the reduced result will be propagate back to the first line of the reduce block
       this is used to void bank conflict when accessing the reduced block
    */
    template<int reduce_m,int y_block_bits,int x_block_bits>
    __device__ void reduce_block_with_bp_1616( float buf[ 16 ][ 17 ] );
    
    /* 
       back propagate results, this function is used to spride answers to shared memory
    */
    template<int block_bits>
    __device__ void backprop_x_block_1616(  float buf[ 16 ][ 17 ] );

};

#include "cuda_reduce.cu"
#endif
