#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "../tensor/apex_tensor.h"
#include "../utils/data_set/apex_kyoto_iterator.h"
#include "../utils/task/apex_tensor_update_task.h"
#include "../crbm/apex_crbm_trainer.h"

using namespace apex_utils;

int main( int argc, char *argv[] ){
	apex_tensor::init_tensor_engine_cpu( 10 );
    KyotoIterator<apex_tensor::GTensor4D,apex_tensor::GTensor4D> itr;        
    apex_rbm::CRBMTrainer<apex_tensor::GTensor4D> trainer;
	TensorUpdateTask<apex_tensor::GTensor4D> tsk( &trainer, &itr );	
    return run_task( argc, argv, &tsk );
}

