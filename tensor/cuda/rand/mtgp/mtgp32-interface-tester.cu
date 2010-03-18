#include "mtgp32-interface.cu"
using namespace mtgp;

/**
 * This function is used to compare the outputs with C program's.
 * @param[in] array data to be printed.
 * @param[in] size size of array.
 * @param[in] block number of blocks.
 */
void print_float_array(const float array[], int size, int block) {
    int b = size / block;

    for (int j = 0; j < 5; j += 5) {
	printf("%.10f %.10f %.10f %.10f %.10f\n",
	       array[j], array[j + 1],
	       array[j + 2], array[j + 3], array[j + 4]);
    }
    for (int i = 1; i < block; i++) {
	for (int j = -5; j < 5; j += 5) {
	    printf("%.10f %.10f %.10f %.10f %.10f\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2],
		   array[b * i + j + 3],
		   array[b * i + j + 4]);
	}
    }
    for (int j = -5; j < 0; j += 5) {
	printf("%.10f %.10f %.10f %.10f %.10f\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2],
	       array[size + j + 3],
	       array[size + j + 4]);
    }
}

/**
 * This function is used to compare the outputs with C program's.
 * @param[in] array data to be printed.
 * @param[in] size size of array.
 * @param[in] block number of blocks.
 */
void print_uint32_array(uint32_t array[], int size, int block) {
    int b = size / block;

    for (int j = 0; j < 5; j += 5) {
	printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
	       " %10" PRIu32 " %10" PRIu32 "\n",
	       array[j], array[j + 1],
	       array[j + 2], array[j + 3], array[j + 4]);
    }
    for (int i = 1; i < block; i++) {
	for (int j = -5; j < 5; j += 5) {
	    printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
		   " %10" PRIu32 " %10" PRIu32 "\n",
		   array[b * i + j],
		   array[b * i + j + 1],
		   array[b * i + j + 2],
		   array[b * i + j + 3],
		   array[b * i + j + 4]);
	}
    }
    for (int j = -5; j < 0; j += 5) {
	printf("%10" PRIu32 " %10" PRIu32 " %10" PRIu32
	       " %10" PRIu32 " %10" PRIu32 "\n",
	       array[size + j],
	       array[size + j + 1],
	       array[size + j + 2],
	       array[size + j + 3],
	       array[size + j + 4]);
    }
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] d_status kernel I/O data.
 * @param[in] num_data number of data to be generated.
 */
void make_uint32_random(mtgp32_kernel_status_t* d_status, int num_data) {
    uint32_t* d_data;
    uint32_t* h_data;
    cudaError_t e;


    printf("generating 32-bit unsigned random numbers.\n");
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data));
    h_data = (uint32_t *) malloc(sizeof(uint32_t) * num_data);
    if (h_data == NULL) {
        printf("failure in allocating host memory for output data.\n");
        exit(1);
    }
    if (cudaGetLastError() != cudaSuccess) {
        printf("error has been occured before kernel call.\n");
        exit(1);
    }

    /* kernel call */
    mtgp32_uint32_kernel<<< BLOCK_NUM, THREAD_NUM>>>( d_status, d_data, num_data / BLOCK_NUM);
    cudaThreadSynchronize();
    
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
        exit(1);
    }
    CUDA_SAFE_CALL(
                   cudaMemcpy(h_data,
                              d_data,
                              sizeof(uint32_t) * num_data,
                              cudaMemcpyDeviceToHost));
    
    print_uint32_array(h_data, num_data, BLOCK_NUM);
    
    printf("generated numbers: %d\n", num_data);
    //free memories
    free(h_data);
    CUDA_SAFE_CALL(cudaFree(d_data));
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param[in] d_status kernel I/O data.
 * @param[in] num_data number of data to be generated.
 */
void make_single_random(mtgp32_kernel_status_t* d_status, int num_data) {
    uint32_t* d_data;
    float* h_data;
    cudaError_t e;

    printf("generating single precision floating point random numbers.\n");
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data));

    h_data = (float *) malloc(sizeof(float) * num_data);
    if (h_data == NULL) {
        printf("failure in allocating host memory for output data.\n");
        exit(1);
    }

    if (cudaGetLastError() != cudaSuccess) {
        printf("error has been occured before kernel call.\n");
        exit(1);
    }

    /* kernel call */
    mtgp32_single_kernel<<< BLOCK_NUM, THREAD_NUM >>>(
	d_status, d_data, num_data / BLOCK_NUM);
    cudaThreadSynchronize();

    e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
        exit(1);
    }

    CUDA_SAFE_CALL(
	cudaMemcpy(h_data,
		   d_data,
		   sizeof(uint32_t) * num_data,
		   cudaMemcpyDeviceToHost));


    print_float_array(h_data, num_data, BLOCK_NUM);
    printf("generated numbers: %d\n", num_data);

    //free memories
    free(h_data);
    CUDA_SAFE_CALL(cudaFree(d_data));
}

int main(int argc, char** argv)
{
    // LARGE_SIZE is a multiple of 16
    int num_data = 10000000;
    int num_unit = LARGE_SIZE * BLOCK_NUM;
    int r;
    mtgp32_kernel_status_t* d_status;

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_status,
			      sizeof(mtgp32_kernel_status_t) * BLOCK_NUM));
    if (argc >= 2) {
        errno = 0;
        num_data = strtol(argv[1], NULL, 10);
        if (errno) {
            printf("%s number_of_output\n", argv[0]);
            return 1;
	}
    } else {
        printf("%s number_of_output\n", argv[0]);
        return 1;
    }
    r = num_data % num_unit;
    if (r != 0) {
        num_data = num_data + num_unit - r;
    }
    make_constant(mtgp32_params_fast_23209);
    make_kernel_data(d_status, mtgp32_params_fast_23209);
    make_uint32_random(d_status, num_data);
    make_single_random(d_status, num_data);

    //finalize
    CUDA_SAFE_CALL(cudaFree(d_status));
    return 0;
}

