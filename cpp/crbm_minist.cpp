#define _CRT_SECURE_NO_WARNINGS

#include "../tensor/apex_tensor.h"
#include "../utils/data_set/apex_minist_iterator.h"
#include "../utils/task/apex_tensor_update_task.h"
#include "../crbm/apex_crbm_trainer.h"

using namespace apex_utils;

int main( int argc, char *argv[] ){
	apex_tensor::init_tensor_engine_cpu( 10 );
    MINISTIterator<apex_tensor::CTensor4D> itr;        
    apex_rbm::CRBMTrainer trainer;
	TensorUpdateTask<apex_tensor::CTensor4D> tsk( &trainer, &itr );	
    return run_task( argc, argv, &tsk );
}

