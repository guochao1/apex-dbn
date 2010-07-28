#include <cstdio>
#include <iostream>
#include "../utils/apex_utils.h"
#include "../tensor/apex_tensor.h"
#include "../utils/apex_config.h"
#include "../cfrbm/apex_cfrbm_model.h"
#include "../cfrbm/apex_cfrbm_adapter.cpp"
#include "../cfrbm/apex_cfrbm.h"
using namespace std;

int main( int argc, char *argv[] ){
    
	apex_tensor::init_tensor_engine_cpu( 10 );
   	FILE *training_file = apex_utils::fopen_check( argv[3], "r" );
	FILE *model_file = apex_utils::fopen_check( argv[4], "wb" );
	
	apex_rbm::CFSRBMTrainParam train_para;
	apex_utils::ConfigIterator train_para_cfg( argv[1] );  
	while( train_para_cfg.next() ){
		   train_para.set_param( train_para_cfg.name(), train_para_cfg.val() );
	}


	apex_rbm::CFSRBMModelParam model_para;
	apex_utils:: ConfigIterator model_para_cfg( argv[2] );
	while( model_para_cfg.next()){
			model_para.set_param( model_para_cfg.name(), model_para_cfg.val() );
	}
	cout << "init done" << endl;	
	vector<apex_tensor::CSTensor2D> training_data = apex_rbm::adapt( training_file ); 
	cout << "adapt done" << endl;	
	apex_rbm::CFSRBMModel model( model_para );
	cout << "set model done"<< endl;
	apex_rbm::CFSRBM *cfsrbm = apex_rbm::factory::create_cfrbm( model, train_para );
	cout << "create_cfrbm done" << endl;
	cfsrbm->train_update_trunk( training_data );
	cout << "update_trunk done" << endl;
	cfsrbm->generate_model( model_file );

}
