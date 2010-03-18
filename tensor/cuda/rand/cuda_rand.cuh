#ifndef _CUDA_RAND_CUH_
#define _CUDA_RAND_CUH_

namespace cuda_rand{
    // note the functions here are not thread safe 

    /* initialize the random module, must be called at begining of system */
    inline void rand_init();

    /* generate single array with size num random float uniform in [1,2) */
    inline const float *rand_singles( size_t num );

    /* 
       get random number from rnd, this method contains the routine to check conflict
       reading in the debug mode
     */
    inline __device__ float get_rand( const float *rnd, int idx );
    
    /* 
       destroy the random module, must be called at end of system
    */
    inline void rand_destroy();
};

#include "cuda_rand_mtgp32.cu"
#endif

