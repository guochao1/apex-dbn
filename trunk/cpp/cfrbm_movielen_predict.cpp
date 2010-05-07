#include <cstdio>
#include <iostream>
#include "../tensor/apex_tensor.h"
#include "../utils/apex_utils.h"
#include "../cfrbm/apex_cfrbm_model.h"
#include "../cfrbm/apex_cfrbm_adapter.cpp"
#include "../cfrbm/apex_cfrbm_predict.cpp"

using namespace std;
int main( int argc, char *argv[] ){

	FILE *model_file = apex_utils::fopen_check( argv[1], "r" );	
   	FILE *training_file = apex_utils::fopen_check( argv[2], "r" );
   	FILE *predict_file = apex_utils::fopen_check( argv[3], "r" );
   	FILE *result = fopen( argv[4], "w" );
	vector<apex_tensor::CSTensor2D> training_data = apex_rbm::adapt( training_file ); 

	apex_rbm::PREDModel model( model_file );

	apex_rbm::Predictor *pred = apex_rbm::factory::create_predictor( model, training_data );

	int user_id, movie_id, rate;
	char other[20];
	while(fscanf(predict_file, "%d\t%d\t%d\t%s\n", &user_id, &movie_id, &rate, other ) != -1){
			apex_tensor::TENSOR_FLOAT predrate = pred->predict( user_id, movie_id );
			fprintf(result, "%d\t%d\t%d\t%f\n", user_id, movie_id, rate, predrate);
	}

	return 0;

}
