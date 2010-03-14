// this is an example tester program 
// about how to use tensor and how to write test programs
// tqchen
#include "../apex_tensor.h"

using namespace apex_tensor;

inline void test(){
    // define a ts[4][5];
    TTensor2D ts(4,5);
    // allocate space
    tensor::alloc_space( ts );

    // assign ts to 1.0f
    ts = 1.0f;
    ts = ts*2.0f + ts*3.0f;

    //should be 5
    printf("%f\n", ts[3][4]);
    
    tensor::free_space( ts );    
}

int main( void ){
    // initalize the tensor engine with random seed 0
    init_tensor_engine( 0 );
    test();
    // destroy the tensor engine on exit
    destroy_tensor_engine();
    return 0;
} 
