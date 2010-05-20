/**
 * @file mtgp32-interface.cu  modified from file mtgp32-cuda.cu
 *
 * @brief Sample Program for CUDA 2.2
 *
 * MTGP32-23209
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>23209</sup>-1.
 *
 * This also generates single precision floating point numbers
 * uniformly distributed in the range [1, 2). (float r; 1.0 <= r < 2.0)
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (Hiroshima University)
 *
 * Copyright (C) 2009 Mutsuo Saito, Makoto Matsumoto and
 * Hiroshima University. All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */
#define __STDC_FORMAT_MACROS 1
#define __STDC_CONSTANT_MACROS 1
extern "C" {
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
}

#ifdef _MSC_VER
#include "stdint.h"
#include "inttypes.h"
#else
#include <stdint.h>
#include <inttypes.h>
#endif


extern "C" {
#include "mtgp32-fast.h"
}

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(X) X
#endif
 
/* we add namespace to avoid conflict*/
namespace mtgp{
    const int MEXP = 23209;
    const int N = 726;
    const int THREAD_NUM=512;
    const int LARGE_SIZE = (THREAD_NUM * 3);
    const int BLOCK_NUM = 128;	     /* You can change this value up to 128 */
    const int TBL_SIZE = 16;


    /**
     * kernel I/O
     * This structure must be initialized before first use.
     */
    struct mtgp32_kernel_status_t {
        uint32_t status[N];
    };
};

/*
 * Generator Parameters.
 */
__constant__ uint32_t __mtgp_param_tbl[mtgp::BLOCK_NUM][mtgp::TBL_SIZE];
__constant__ uint32_t __mtgp_temper_tbl[mtgp::BLOCK_NUM][mtgp::TBL_SIZE];
__constant__ uint32_t __mtgp_single_temper_tbl[mtgp::BLOCK_NUM][mtgp::TBL_SIZE];
__constant__ uint32_t __mtgp_pos_tbl[mtgp::BLOCK_NUM];
__constant__ uint32_t __mtgp_sh1_tbl[mtgp::BLOCK_NUM];
__constant__ uint32_t __mtgp_sh2_tbl[mtgp::BLOCK_NUM];

/* high_mask and low_mask should be set by make_constant(), but
 * did not work.
 */
__constant__ uint32_t __mtgp_mask = 0xff800000;

/**
 * Shared memory
 * The generator's internal status vector.
 */
__shared__ uint32_t __mtgp_status[mtgp::LARGE_SIZE]; /* 512 * 3 elements, 6144 bytes. */
namespace mtgp{
    /**
     * The function of the recursion formula calculation.
     *
     * @param[in] X1 the farthest part of state array.
     * @param[in] X2 the second farthest part of state array.
     * @param[in] Y a part of state array.
     * @param[in] bid block id.
     * @return output
     */
    inline __device__ uint32_t para_rec(uint32_t X1, uint32_t X2, uint32_t Y, int bid) {
        uint32_t X = (X1 & __mtgp_mask) ^ X2;
        uint32_t MAT;
        
        X ^= X << __mtgp_sh1_tbl[bid];
        Y = X ^ (Y >> __mtgp_sh2_tbl[bid]);
        MAT = __mtgp_param_tbl[bid][Y & 0x0f];
        return Y ^ MAT;
    }
    
    /**
     * The tempering function.
     *
     * @param[in] V the output value should be tempered.
     * @param[in] T the tempering helper value.
     * @param[in] bid block id.
     * @return the tempered value.
     */
    inline __device__ uint32_t temper(uint32_t V, uint32_t T, int bid) {
        uint32_t MAT;
        
        T ^= T >> 16;
        T ^= T >> 8;
        MAT = __mtgp_temper_tbl[bid][T & 0x0f];
        return V ^ MAT;
    }
    
    /**
     * The tempering and converting function.
     * By using the preset-ted table, converting to IEEE format
     * and tempering are done simultaneously.
     *
     * @param[in] V the output value should be tempered.
     * @param[in] T the tempering helper value.
     * @param[in] bid block id.
     * @return the tempered and converted value.
     */
    inline __device__ uint32_t temper_single(uint32_t V, uint32_t T, int bid) {
        uint32_t MAT;
        uint32_t r;
        
        T ^= T >> 16;
        T ^= T >> 8;
        MAT = __mtgp_single_temper_tbl[bid][T & 0x0f];
        r = (V >> 9) ^ MAT;
        return r;
    }
    
    /**
     * Read the internal state vector from kernel I/O data, and
     * put them into shared memory.
     *
     * @param[out] __mtgp_status shared memory.
     * @param[in] d_status kernel I/O data
     * @param[in] bid block id
     * @param[in] tid thread id
     */
    inline __device__ void __mtgp_status_read(uint32_t __mtgp_status[LARGE_SIZE],
                                              const mtgp32_kernel_status_t *d_status,
                                              int bid,
                                              int tid) {
        __mtgp_status[LARGE_SIZE - N + tid] = d_status[bid].status[tid];
        if (tid < N - THREAD_NUM) {
            __mtgp_status[LARGE_SIZE - N + THREAD_NUM + tid]
                = d_status[bid].status[THREAD_NUM + tid];
        }
        __syncthreads();
    }
    
    /**
     * Read the internal state vector from shared memory, and
     * write them into kernel I/O data.
     *
     * @param[out] d_status kernel I/O data
     * @param[in] __mtgp_status shared memory.
     * @param[in] bid block id
     * @param[in] tid thread id
     */
    inline __device__ void __mtgp_status_write(mtgp32_kernel_status_t *d_status,
                                               const uint32_t __mtgp_status[LARGE_SIZE],
                                               int bid,
                                               int tid) {
        d_status[bid].status[tid] = __mtgp_status[LARGE_SIZE - N + tid];
        if (tid < N - THREAD_NUM) {
            d_status[bid].status[THREAD_NUM + tid]
                = __mtgp_status[4 * THREAD_NUM - N + tid];
        }
        __syncthreads();
    }
    
    /**
     * kernel function.
     * This function generates 32-bit unsigned integers in d_data
     *
     * @param[in,out] d_status kernel I/O data
     * @param[out] d_data output
     * @param[in] size number of output data requested.
     */
    __global__ void mtgp32_uint32_kernel(mtgp32_kernel_status_t* d_status,
                                         uint32_t* d_data, int size) {
        const int bid = blockIdx.x;
        const int tid = threadIdx.x;
        int pos = __mtgp_pos_tbl[bid];
        uint32_t r;
        uint32_t o;
        
    // copy __mtgp_status data from global memory to shared memory.
        __mtgp_status_read(__mtgp_status, d_status, bid, tid);
        
        // main loop
        for (int i = 0; i < size; i += LARGE_SIZE) {
            
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
            if ((i == 0) && (bid == 0) && (tid <= 1)) {
                printf("__mtgp_status[LARGE_SIZE - N + tid]:%08x\n",
                       __mtgp_status[LARGE_SIZE - N + tid]);
                printf("__mtgp_status[LARGE_SIZE - N + tid + 1]:%08x\n",
                       __mtgp_status[LARGE_SIZE - N + tid + 1]);
                printf("__mtgp_status[LARGE_SIZE - N + tid + pos]:%08x\n",
                       __mtgp_status[LARGE_SIZE - N + tid + pos]);
                printf("sh1:%d\n", __mtgp_sh1_tbl[bid]);
                printf("sh2:%d\n", __mtgp_sh2_tbl[bid]);
                printf("__mtgp_mask:%08x\n", __mtgp_mask);
                for (int j = 0; j < 16; j++) {
                    printf("tbl[%d]:%08x\n", j, __mtgp_param_tbl[0][j]);
                }
            }
#endif
            r = para_rec(__mtgp_status[LARGE_SIZE - N + tid],
                         __mtgp_status[LARGE_SIZE - N + tid + 1],
                         __mtgp_status[LARGE_SIZE - N + tid + pos],
                         bid);
            __mtgp_status[tid] = r;
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
            if ((i == 0) && (bid == 0) && (tid <= 1)) {
                printf("__mtgp_status[tid]:%08x\n", __mtgp_status[tid]);
            }
#endif
            o = temper(r, __mtgp_status[LARGE_SIZE - N + tid + pos - 1], bid);
#if defined(DEBUG) && defined(__DEVICE_EMULATION__)
            if ((i == 0) && (bid == 0) && (tid <= 1)) {
                printf("r:%08" PRIx32 "\n", r);
            }
#endif
            d_data[size * bid + i + tid] = o;
            __syncthreads();
            r = para_rec(__mtgp_status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
                         __mtgp_status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
                         __mtgp_status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
                         bid);
            __mtgp_status[tid + THREAD_NUM] = r;
            o = temper(r,
                       __mtgp_status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
                       bid);
            d_data[size * bid + THREAD_NUM + i + tid] = o;
            __syncthreads();
            r = para_rec(__mtgp_status[2 * THREAD_NUM - N + tid],
                         __mtgp_status[2 * THREAD_NUM - N + tid + 1],
                         __mtgp_status[2 * THREAD_NUM - N + tid + pos],
		     bid);
            __mtgp_status[tid + 2 * THREAD_NUM] = r;
            o = temper(r, __mtgp_status[tid + pos - 1 + 2 * THREAD_NUM - N], bid);
            d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
            __syncthreads();
        }
        // write back __mtgp_status for next call
        __mtgp_status_write(d_status, __mtgp_status, bid, tid);
    }
    
    /**
     * kernel function.
     * This function generates single precision floating point numbers in d_data.
     *
     * @param[in,out] d_status kernel I/O data
     * @param[out] d_data output. IEEE single precision format.
     * @param[in] size number of output data requested.
     */
    __global__ void mtgp32_single_kernel(mtgp32_kernel_status_t* d_status,
                                         uint32_t* d_data, int size)
    {
        
        const int bid = blockIdx.x;
        const int tid = threadIdx.x;
        int pos = __mtgp_pos_tbl[bid];
        uint32_t r;
        uint32_t o;
    
        // copy __mtgp_status data from global memory to shared memory.
        __mtgp_status_read(__mtgp_status, d_status, bid, tid);
        
        // main loop
        for (int i = 0; i < size; i += LARGE_SIZE) {
            r = para_rec(__mtgp_status[LARGE_SIZE - N + tid],
                         __mtgp_status[LARGE_SIZE - N + tid + 1],
                         __mtgp_status[LARGE_SIZE - N + tid + pos],
                         bid);
            __mtgp_status[tid] = r;
            o = temper_single(r, __mtgp_status[LARGE_SIZE - N + tid + pos - 1], bid);
            d_data[size * bid + i + tid] = o;
            __syncthreads();
            r = para_rec(__mtgp_status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
                         __mtgp_status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
                         __mtgp_status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
                         bid);
            __mtgp_status[tid + THREAD_NUM] = r;
            o = temper_single(
                              r,
                              __mtgp_status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
                              bid);
            d_data[size * bid + THREAD_NUM + i + tid] = o;
            __syncthreads();
            r = para_rec(__mtgp_status[2 * THREAD_NUM - N + tid],
                         __mtgp_status[2 * THREAD_NUM - N + tid + 1],
                         __mtgp_status[2 * THREAD_NUM - N + tid + pos],
                         bid);
            __mtgp_status[tid + 2 * THREAD_NUM] = r;
            o = temper_single(r,
                              __mtgp_status[tid + pos - 1 + 2 * THREAD_NUM - N],
                              bid);
            d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
            __syncthreads();
        }
        // write back __mtgp_status for next call
        __mtgp_status_write(d_status, __mtgp_status, bid, tid);
    }
    
    /**
     * This function sets constants in device memory.
     * @param[in] params input, MTGP32 parameters.
     */
    void make_constant(const mtgp32_params_fast_t params[]) {
        const int size1 = sizeof(uint32_t) * BLOCK_NUM;
        const int size2 = sizeof(uint32_t) * BLOCK_NUM * mtgp::TBL_SIZE;
        uint32_t *h___mtgp_pos_tbl;
        uint32_t *h___mtgp_sh1_tbl;
        uint32_t *h___mtgp_sh2_tbl;
        uint32_t *h___mtgp_param_tbl;
        uint32_t *h___mtgp_temper_tbl;
        uint32_t *h___mtgp_single_temper_tbl;
#if 0
        uint32_t *h___mtgp_mask;
#endif
        h___mtgp_pos_tbl = (uint32_t *)malloc(size1);
        h___mtgp_sh1_tbl = (uint32_t *)malloc(size1);
        h___mtgp_sh2_tbl = (uint32_t *)malloc(size1);
        h___mtgp_param_tbl = (uint32_t *)malloc(size2);
        h___mtgp_temper_tbl = (uint32_t *)malloc(size2);
        h___mtgp_single_temper_tbl = (uint32_t *)malloc(size2);
#if 0
        h___mtgp_mask = (uint32_t *)malloc(sizeof(uint32_t));
#endif
        if (h___mtgp_pos_tbl == NULL
            || h___mtgp_sh1_tbl == NULL
            || h___mtgp_sh2_tbl == NULL
            || h___mtgp_param_tbl == NULL
            || h___mtgp_temper_tbl == NULL
            || h___mtgp_single_temper_tbl == NULL
#if 0
            || h___mtgp_mask == NULL
#endif
            ) {
            printf("failure in allocating host memory for constant table.\n");
            exit(1);
        }
#if 0
        h___mtgp_mask = params[0].__mtgp_mask;
#endif
        for (int i = 0; i < BLOCK_NUM; i++) {
            h___mtgp_pos_tbl[i] = params[i].pos;
            h___mtgp_sh1_tbl[i] = params[i].sh1;
            h___mtgp_sh2_tbl[i] = params[i].sh2;
            for (int j = 0; j < mtgp::TBL_SIZE; j++) {
                h___mtgp_param_tbl[i * mtgp::TBL_SIZE + j] = params[i].tbl[j];
                h___mtgp_temper_tbl[i * mtgp::TBL_SIZE + j] = params[i].tmp_tbl[j];
                h___mtgp_single_temper_tbl[i * mtgp::TBL_SIZE + j] = params[i].flt_tmp_tbl[j];
            }
        }
        // copy from malloc area only
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(__mtgp_pos_tbl, h___mtgp_pos_tbl, size1));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(__mtgp_sh1_tbl, h___mtgp_sh1_tbl, size1));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(__mtgp_sh2_tbl, h___mtgp_sh2_tbl, size1));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(__mtgp_param_tbl, h___mtgp_param_tbl, size2));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(__mtgp_temper_tbl, h___mtgp_temper_tbl, size2));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(__mtgp_single_temper_tbl,
                                          h___mtgp_single_temper_tbl, size2));
#if 0
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(&__mtgp_mask,
                                          &h___mtgp_mask, sizeof(uint32_t)));
#endif
        free(h___mtgp_pos_tbl);
        free(h___mtgp_sh1_tbl);
        free(h___mtgp_sh2_tbl);
        free(h___mtgp_param_tbl);
        free(h___mtgp_temper_tbl);
        free(h___mtgp_single_temper_tbl);
#if 0
        free(h___mtgp_mask);
#endif
}

    /**
     * This function initializes kernel I/O data.
     * @param[out] d_status output kernel I/O data.
     * @param[in] params MTGP32 parameters. needed for the initialization.
     */
    void make_kernel_data(mtgp32_kernel_status_t *d_status,
                          mtgp32_params_fast_t params[]) {
        mtgp32_kernel_status_t* h___mtgp_status = (mtgp32_kernel_status_t *) malloc(
                                                                                                  sizeof(mtgp32_kernel_status_t) * BLOCK_NUM);
        
        if (h___mtgp_status == NULL) {
            printf("failure in allocating host memory for kernel I/O data.\n");
            exit(8);
        }
        for (int i = 0; i < BLOCK_NUM; i++) {
            mtgp32_init_state(&(h___mtgp_status[i].status[0]), &params[i], i + 1);
        }
#if defined(DEBUG)
        printf("h___mtgp_status[0].status[0]:%08"PRIx32"\n", h___mtgp_status[0].status[0]);
        printf("h___mtgp_status[0].status[1]:%08"PRIx32"\n", h___mtgp_status[0].status[1]);
        printf("h___mtgp_status[0].status[2]:%08"PRIx32"\n", h___mtgp_status[0].status[2]);
        printf("h___mtgp_status[0].status[3]:%08"PRIx32"\n", h___mtgp_status[0].status[3]);
#endif
        CUDA_SAFE_CALL(cudaMemcpy(d_status,
			      h___mtgp_status,
                                  sizeof(mtgp32_kernel_status_t) * BLOCK_NUM,
                                  cudaMemcpyHostToDevice));
        free(h___mtgp_status);
    }  
};
