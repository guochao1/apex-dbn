#ifndef _TEST_CMP_CPP_
#define _TEST_CMP_CPP_

#include "../apex_tensor.h"
#include "test_stats.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
using namespace apex_tensor;

const int NUM_ITER = 20000;

const int V_MAX   = 20;
const int H_MAX   = 15;
const int V_Y_MAX = 28;
const int V_X_MAX = 28;
const int F_Y_MAX = 12;
const int F_X_MAX = 12;
const int H_Y_MAX = V_Y_MAX - F_Y_MAX + 1; 
const int H_X_MAX = V_X_MAX - F_X_MAX + 1; 
const TENSOR_FLOAT sd = 1.0f;

#include "test_cmp.h"

int main( void ){
    init_tensor_engine_cpu(0);
    init_tensor_engine_gpu();
	
	test_conv2_r_big_filter( NUM_ITER );
    test_conv2_r_valid( NUM_ITER );
	test_conv2_full( NUM_ITER );

    system("pause");
    return 0;
} 

#endif

