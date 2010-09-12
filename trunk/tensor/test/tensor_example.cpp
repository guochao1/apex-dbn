#include "../apex_tensor.h"
using namespace apex_tensor;

const int Y_MAX = 10;
const int X_MAX = 10;

// this is an example of how to use tensor library
int main( void ){
    // all data starts from CPU
    CTensor2D ca(Y_MAX,X_MAX),cb(Y_MAX,X_MAX),cc(Y_MAX,X_MAX);
    GTensor2D ga(Y_MAX,X_MAX),gb(Y_MAX,X_MAX),gc(Y_MAX,X_MAX);
    // we need to allocate space before use every thing
    // pretty C-style: reference apex_tensor_cpu.h
    tensor::alloc_space( ca );
    tensor::alloc_space( cb );
    tensor::alloc_space( cc );
    // note GTensors allocate space in GPU
    tensor::alloc_space( ga );
    tensor::alloc_space( gb );
    tensor::alloc_space( gc );
    
    /* copy data to GPU if you want */
    tensor::copy( ga, ca );
    tensor::copy( gb, cb );
    // calculation can be carried in GPU
    // basic operations
    gc = ga * 10.0f + gb * 2.0f; 
    // matrix multiplication   
    gc = dot( ga, gb );
    gc = dot( ga.T(), gb );
    // calculation can also be carried in CPU
    cc = dot( ca, cb );
    // and simple 2D convoltion, note full means 'full' option in matlab's 2D 
    // convolution
    tensor::crbm::conv2_full( gc, ga, gb );  

    // copy back data if you want
    tensor::copy( cc, gc );    
    // free space if they are no longer useful
    tensor::free_space( ca );
    tensor::free_space( cb );
    tensor::free_space( cc );
    tensor::free_space( ga );
    tensor::free_space( gb );
    tensor::free_space( gc );
    return 0;
}
