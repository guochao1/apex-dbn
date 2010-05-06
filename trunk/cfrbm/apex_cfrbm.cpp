#ifndef _APEX_CFSRBM_CPP_
#define _APEX_CFSRBM_CPP_

#include "apex_cfrbm.h"
#include "apex_cfrbm_model.h"
#include "../tensor/apex_tensor.h"
#include "../tensor/apex_tensor_sparse.h"
#include <vector>
#include <iostream>
namespace apex_rbm{
    using namespace std;
    using namespace apex_tensor;

    // bianry node
    class CFSRBMBinaryNode {
    public:
        virtual void sample  ( TTensor1D &state, const TTensor1D &mean ){
            state = sample_binary( mean );
        }
        virtual void cal_mean( TTensor1D &mean, const TTensor1D &energy){
            mean = sigmoid( energy );
        }        
    };

	class SoftMaxNode {
	public:
		virtual void sample	( TSTensor2D &state, const TSTensor2D &mean ){
			tensor::cf_rbm::sample_softmax( state,  mean );
		}
		virtual void cal_mean(TSTensor2D &mean, const TSTensor2D &energy ){
			mean =  energy ;
			tensor::cf_rbm::norm_softmax( mean );
		}	
	};	
    // simple implementation of srbm
    class CFSRBMSimple: public CFSRBM{
    private:
        CFSRBMTrainParam param;
		int sample_counter;
	private:
	    TTensor1D 	h_bias;
		TTensor2D	v_bias;
        TTensor3D 	W;
    private:
        TTensor1D 	d_h_bias;
		TTensor2D	d_v_bias;
        TTensor3D 	d_W;
    private:
        TTensor1D 	h_neg, h_pos;
		TSTensor2D	v_neg, v_pos;
	private:
		CFSRBMBinaryNode *h_node;
		SoftMaxNode *v_node;
    private:
        // initalize the space
        inline void init( const CFSRBMModel &model ){
			h_bias	 = clone( model.h_bias );
			v_bias	 = clone( model.v_bias );
			W		 = clone( model.W );	
            d_h_bias = clone( model.d_h_bias );
            d_v_bias = clone( model.d_v_bias );
            d_W      = clone( model.d_W );
            h_neg    = alloc_like( model.d_h_bias );
            h_pos    = alloc_like( model.d_h_bias );
        }
    public:
        CFSRBMSimple( const CFSRBMModel &model, const CFSRBMTrainParam &param ){
            init( model );
            this->param = param;
			this->h_node = new CFSRBMBinaryNode();
			this->v_node = new SoftMaxNode();
            // intialize the tensor engine
            init_tensor_engine( 0 );
        }
        
        // deallocate the space
        virtual ~CFSRBMSimple(){
			tensor::free_space( h_bias );
			tensor::free_space( v_bias );
			tensor::free_space( W );
            tensor::free_space( d_h_bias );
            tensor::free_space( d_v_bias );
            tensor::free_space( d_W );
            tensor::free_space( v_pos );
            tensor::free_space( v_neg );
            tensor::free_space( h_pos );
            tensor::free_space( h_neg );

            // destroy the tensor engine
            destroy_tensor_engine();
        }              
    private:

		inline void upfeed_soft_max( TSTensor2D &soft_max, TTensor1D &h_pos, TTensor3D &W, TTensor1D &h_bias ){
			for(int i = 0; i < soft_max.y_max; ++ i){
					TSTensor1D line = soft_max[ i ];
					tensor::sadd__dot( h_pos, line, W[ i ] );
			}
			h_pos += h_bias;
		}

		inline void downfeed_soft_max( TTensor1D &h_neg, TSTensor2D &soft_max, TTensor3D &W, TTensor2D &v_bias ){
			for(int i = 0; i < soft_max.y_max; ++ i){
					TSTensor1D line = soft_max[ i ];
					line = dot( h_neg, W[ i ].T() );
					tensor::sadd( line, v_bias[ i ] );
			}
		}

        // calculate the datas in cd steps	
        inline void cal_cd_steps( TSTensor2D &v_pos, TSTensor2D &v_neg,
                                  TTensor1D &h_pos, TTensor1D &h_neg ){

            // go up
			upfeed_soft_max( v_pos, h_pos, W, h_bias ); 
            h_node->cal_mean( h_pos, h_pos );
            // negative steps
            for( int i = 0 ; i < param.cd_step ; i ++ ){

                // sample h
                h_node->sample( h_neg, h_neg );

                // go down
				downfeed_soft_max( h_neg, v_neg, W, v_bias );
                v_node->cal_mean( v_neg, v_neg );
                v_node->sample  ( v_neg, v_neg );
				
                // go up
				upfeed_soft_max( v_neg, h_neg, W, h_bias );
                h_node->cal_mean( h_neg, h_neg );
            }
        }
	        // update the weight of the last layer
        inline void update_weight(){

            const float eta = param.learning_rate/param.batch_size;
            
            if( param.chg_hidden_bias ){
                h_bias = h_bias * ( 1-eta*param.wd_h ) + d_h_bias * eta;
                d_h_bias*= param.momentum;
            }
            if( param.chg_visible_bias ){
                v_bias = v_bias * ( 1-eta*param.wd_v ) + d_v_bias * eta;
                d_v_bias*= param.momentum;
            }

            W   = W * ( 1-eta*param.wd_W ) + d_W * eta;            
            d_W *= param.momentum;
		}

        // update in training
        inline void train_update(){

            cal_cd_steps( v_pos, v_neg,  h_pos, h_neg );
//			cout << "cal_cd_steps done\n";
            // calculate the gradient
			for( int i = 0; i < d_W.z_max; ++ i ){
				TTensor2D line = d_W[ i ];
				tensor::sadd__dot_lt( line, v_pos[ i ], h_pos );
				tensor::ssub__dot_lt( line, v_pos[ i ], h_neg );
			}
//			cout << "update d_W done\n";
            if( param.chg_hidden_bias ){
				d_h_bias += h_pos;
                d_h_bias -= h_neg; 
            }
//			cout << "update d_h_bias done\n";
            if( param.chg_visible_bias ){
				tensor::add( d_v_bias, d_v_bias, v_pos ); 
				tensor::sub( d_v_bias, d_v_bias, v_neg );
            }
//			cout << "update d_v_bias done\n";
            if( ++sample_counter == param.batch_size ){
                update_weight();
				cout << "update_weight done\n";
                sample_counter = 0;
            }
//			cout << "train_update done\n";
        }

		inline void setup_input( const apex_tensor::CSTensor2D &data ){
				tensor::copy( this->v_pos, data);
				tensor::copy( this->v_neg, data);
		}

    public:
        virtual void train_update( const apex_tensor::CSTensor2D &data ){
			this->v_pos = alloc_like( data );
			this->v_neg = alloc_like( data );
//			cout << "alloc_like done\n";
            setup_input( data );
//			cout << "setup_input done\n";
            train_update();
//			cout << "train_update done\n";
			tensor::free_space( this->v_pos );
			tensor::free_space( this->v_neg );
//			cout << "free_space done\n";
        }
        virtual void train_update_trunk( const vector<apex_tensor::CSTensor2D> &data ){
            for( int i = 0 ; i < (int)data.size() ; ++ i ){
				cout << "line\t" << i << endl;
                train_update( data[i] );
			}
        }

		virtual void generate_model(FILE *fo){
				tensor::save_to_file( W, fo );
				tensor::save_to_file( h_bias, fo );
				tensor::save_to_file( v_bias, fo );
		}
    };
    
    namespace factory{

        CFSRBM *create_cfrbm( const CFSRBMModel &model, const CFSRBMTrainParam &param ){
            return new CFSRBMSimple( model, param );
        }

   };
};

#endif

