#include <cstdio>
#include <iostream>
#include "../tensor/apex_tensor.h"
#include "../utils/apex_utils.h"
#include "../cfrbm/apex_cfrbm_model.h"
#include "../cfrbm/apex_cfrbm_adapter.cpp"
#include "../cfrbm/apex_cfrbm_predict.cpp"

using namespace std;
class RMSE{
	int count;
	TENSOR_FLOAT res;
	TENSOR_FLOAT tmp;
	public:
		RMSE():count(0),res(0),tmp(0){
		}
		void update(const TENSOR_FLOAT &predicate, const TENSOR_FLOAT &rate){
			tmp = predicate - rate;
			res += tmp*tmp; 
			count ++;
		}
		void display(){
			res /= count;
			res = sqrt(res);
			res /= 1.6;
			cout << "RMSE\t" << res << endl;
		}
};

class NMAE{
	int count;
	TENSOR_FLOAT res;
	public:
		NMAE():count(0),res(0){
		}
		void update(const TENSOR_FLOAT &predicate, const TENSOR_FLOAT &rate){
			res += abs(predicate - rate); 
			count ++;
		}
		void display(){
			res /= count;
			res /= 1.6;
			cout << "NMAE\t" << res << endl;
		}
};

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
	RMSE rmse;
	NMAE nmae;
	while(fscanf(predict_file, "%d\t%d\t%d\t%s\n", &user_id, &movie_id, &rate, other ) != -1){
			apex_tensor::TENSOR_FLOAT predicate = pred->predict( user_id, movie_id );
			if(predicate <= 0)predicate = 1;
			fprintf(result, "%d\t%d\t%d\t%f\t%f\n", user_id, movie_id, rate, predicate, abs(rate - predicate));
			rmse.update(predicate, rate);
			nmae.update(predicate, rate);

	}
	rmse.display();
	nmae.display();
	return 0;

}
