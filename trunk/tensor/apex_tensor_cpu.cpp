#ifndef _APEX_TENSOR_CPU_CPP_
#define _APEX_TENSOR_CPU_CPP_

#include "apex_tensor.h"
#include "../external/apex_random.h"
#include <cmath>
#include <cstring>

// preserve for possible latter use
// tqchen
namespace apex_tensor{

    // initialize function and deconstructor
    static int init_counter = 0;
    // intialize the tensor engine for use, seed is 
    // the seed for random number generator    
    void init_tensor_engine_cpu( int seed ){
        if( init_counter ++ == 0 ){
            apex_random::seed( seed );
        }
    }
    // this function is called when the program exits
    void destroy_tensor_engine_cpu(){
        // do nothing
    }    
};
#endif
