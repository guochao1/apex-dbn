#include <cstdio>
#include "../utils/apex_config.h"
#include "../cfrbm/apex_cfrbm_model.h"
#include "../cfrbm/apex_cfrbm_adapter.cpp"
#include "../cfrbm/apex_cfrbm.cpp"

int main( int argc, char *argv[] ){

   	FILE *training_file = fopen( argv[2], "r" );	
	FILE *model_file = fopen( argv[3], "wb" );
	
	apex_rbm::CFSRBMTrainParam train_para;
	apex_utils::ConfigIterator train_para_cfg( argv[0] );  
	while( train_para_cfg.next() ){
		   train_para.set_param( train_para_cfg.name(), train_para_cfg.val() );
	}

	apex_rbm::CFSRBMModelParam model_para;
	apex_utils:: ConfigIterator model_para_cfg( argv[1] );
	while( model_para_cfg.next()){
			model_para.set_param( model_para_cfg.name(), model_para_cfg.val() );
	}
			
	vector<apex_tensor::CSTensor2D> training_data = apex_rbm::adapt( training_file ); 

	apex_rbm::CFSRBMModel model( model_para );

	apex_rbm::CFSRBM *cfsrbm = apex_rbm::factory::create_cfrbm( model, train_para );

	cfsrbm->train_update_trunk( training_data );

	cfsrbm->generate_model( model_file );

}
