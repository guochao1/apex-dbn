#ifndef _APEX_CFSRBM_PRED_CPP_
#define _APEX_CFSRBM_PRED_CPP_

#include "../tensor/apex_tensor.h"
#include "../tensor/apex_tensor_sparse.h"
#include "apex_cfrbm_model.h"
#include <vector>

namespace apex_rbm{
	using namespace std;
	using namespace apex_tensor;

	class Predictor{

		private:
			vector<TTensor1D> pred;
			TTensor1D h_bias;
			TTensor2D v_bias;
			TTensor3D W;
			int size_softmax;
		private:
			inline void feed_forward( TSTensor2D &soft_max, TTensor1D &param, TTensor3D &W, TTensor1D &h_bias ){
					for(int i = 0; i < soft_max.y_max; ++ i){
						TSTensor1D line = soft_max[ i ];
						tensor::sadd__dot( param, line, W[ i ] );
					}
					param += h_bias;
			}
		public:
			Predictor( const PREDModel &model, const vector<TSTensor2D> train ){
					h_bias = clone( model.h_bias );
					v_bias = clone( model.v_bias );
					W = clone( model.W )
					size_softmax = train[ 0 ];
					for(int i = 0; i < (int)train.size(); ++ i){
						TTensor1D param = alloc_like(model.h_bias);
						feed_forward( train[ i ], param, W, h_bias );
						pred.push_back( param );
					}
			}
			inline TENSOR_FLOAT predict( int x, int y ){
					TENSOR_FLOAT sum = 0, expect = 0;
					for( int i = 0; i < size_softmax; ++ i ){
						TTensor1D param = pred[ x - 1 ];
						TENSOR_FLOAT tmp = 1;
						for( int j = 0; j < param.x_max; ++ j)
							tmp *= ( 1 + exp( param[ j ] + W[ i ][ y ][ j ] ));
						sum += tmp;
						expect += ( i + 1 )*tmp;
					}
					return expect/sum;
			}
			~CFSRBMSimple(){
				tensor::free_space( h_bias );
				tensor::free_space( v_bias );
				tensor::free_space( W );
        	}
	};

	namespace factory{

		Predictor *create_predictor( const PREDModel &model, const vector<CSTensor2D> train ){
				return new Predictor( model, train );
		}

	};

};	
#endif
