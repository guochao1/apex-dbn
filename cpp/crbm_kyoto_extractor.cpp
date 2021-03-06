#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "../tensor/apex_tensor.h"
#include "../utils/data_set/apex_kyoto_iterator.h"
#include "../utils/task/apex_tensor_update_task.h"
#include "../crbm/crbm_simple/apex_crbm_extractor.h"

using namespace apex_utils;
using namespace apex_utils::deprecated;

int main( int argc, char *argv[] ){
	apex_tensor::init_tensor_engine_cpu( 10 );
    KyotoIterator<apex_tensor::CTensor4D,apex_tensor::CTensor4D> itr;        
    apex_rbm::CRBMExtractor ext;
	TensorUpdateTask<apex_tensor::CTensor4D> tsk( &ext, &itr );	
    return run_task( argc, argv, &tsk );
}

