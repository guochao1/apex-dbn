#include <cstdio>
#include <iostream>
#include "../tensor/apex_tensor.h"
#include "../utils/apex_utils.h"
#include "../cfrbm/apex_cfrbm_model"
#include "../cfrbm/apex_cfrbm_predict.cpp"

using namespace std;
int main( int argc, char *argv[] ){

	FILE *model_file = apex_utils::fopen_check( argv[1], "r" );	
   	FILE *training_file = apex_utils::fopen_check( argv[2], "r" );
   	FILE *predict_file = apex_utils::fopen_check( argv[3], "r" );
	ofstream fout ( argv[4] );

	vector<apex_tensor::CSTensor2D> training_data = apex_rbm::adapt( training_file ); 

	apex_rbm::PREDModel modle( model_file );

	Predictor *pred = apex_rbm::factory::create_predictor( model, training_data );

	int user_id, movie_id, rate;
	char other[20];
	while(fscanf(f, "%d\t%d\t%d\t%s\n", &user_id, &movie_id, &rate, other ) != -1){
			TENSOR_FLOAT predrate = pred->predict( user_id, movie_id );
			fout << user_id << "\t" << movie_id << "\t" << rate << "\t" << predrate << endl;
	}

	return 0;

}
