#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "../crbm/crbm_light/apex_crbm_train.h"

int main( int argc, char *argv[] ){
	apex_tensor::init_tensor_engine_cpu( 10 );
    apex_rbm::CRBMLightTrainTask tsk;    
    return run_task( argc, argv, &tsk );
}

