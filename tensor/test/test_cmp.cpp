#ifndef _TEST_CMP_CPP_
#define _TEST_CMP_CPP_

#include "../apex_tensor.h"
#include "test_stats.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
using namespace apex_tensor;

const int NUM_ITER = 10;

const int NUM_SAMPLE = 100;
const int POOL_SIZE = 3;
const int V_MAX   = 10;
const int H_MAX   = 24;
const int V_Y_MAX = 500;
const int V_X_MAX = 500;
const int F_Y_MAX = 10;
const int F_X_MAX = 10;
const int H_Y_MAX = V_Y_MAX - F_Y_MAX + 1; 
const int H_X_MAX = V_X_MAX - F_X_MAX + 1; 
const int P_Y_MAX = H_Y_MAX / POOL_SIZE;
const int P_X_MAX = H_X_MAX / POOL_SIZE;

const TENSOR_FLOAT sd = 1.0f;

#include "test_cmp.h"

int main( void ){
    init_tensor_engine_cpu(0);
    init_tensor_engine(0);
    //    test_norm_maxpooling_2D( NUM_ITER );
        test_pool_up( NUM_ITER );
          test_add_sparse_info( NUM_ITER );
    //    test_sample_maxpooling_2D( NUM_ITER, NUM_SAMPLE );
    //    test_sum_2D( NUM_ITER );
    //test_sum_2DX( NUM_ITER );        
    //test_conv2_r_valid( NUM_ITER );
    
    //    test_conv2_full( NUM_ITER );
    //test_conv2_r_valid( NUM_ITER );
    //    test_conv2_r_big_filter( NUM_ITER );
       //    test_gaussian( NUM_ITER ); 
       //	test_refill_edge_area( NUM_ITER );
       //    test_sadd__scale( NUM_ITER );
    /*
	


    


    

    
	test_conv2_r_big_filter( NUM_ITER );

    */
    
    destroy_tensor_engine();
    return 0;
} 

#endif

