#define _CRT_SECURE_NO_WARNINGS

#include "../tensor/apex_tensor.h"
#include "../utils/data_set/apex_minist_iterator.h"
#include "../utils/task/apex_tensor_update_task.h"
#include "../srbm/apex_srbm_trainer.h"

using namespace apex_utils;

int main( int argc, char *argv[] ){
	apex_tensor::init_tensor_engine_cpu( 10 );
    MINISTIterator itr;        
    apex_rbm::SRBMTrainer trainer;
	Tensor1DUpdateTask tsk( &trainer, &itr );	
    return run_task( argc, argv, &tsk );
}

