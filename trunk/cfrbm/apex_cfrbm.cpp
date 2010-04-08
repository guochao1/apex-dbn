#ifndef _APEX_CFRBM_CPP_
#define _APEX_CFRBM_CPP_

#include "apex_cfrbm.h"
#include "apex_cfrbm_model.h"
#include "apex_cfrbm_model_stats.h"
#include "../tensor/apex_tensor.h"
#include "../tensor/apex_tensor_sparse.h"

#include <vector>

namespace apex_cfrbm{
    using namespace std;
    using namespace apex_tensor;

    //node of SRBM
    class CFSRBMNode{
    public:
        virtual void sample  ( STTensor1D &state, const STTensor1D &mean ) = 0;
        virtual void sample  ( TTensor1D &state, const TTensor1D &mean ) = 0;
        virtual void sample  ( STTensor2D &state, const STTensor2D &mean ) = 0;
        virtual void cal_mean( STTensor1D &mean , const STTensor1D &energy) = 0;
        virtual void cal_mean( STTensor2D &mean , const STTensor2D &energy) = 0;
    };
    // bianry node
    class CFSRBMBinaryNode : public CFSRBMNode{
    public:
        virtual void sample  ( STTensor1D &state, const STTensor1D &mean ){
            state = sample_binary( mean );
        }
		virtual void sample ( TTensor1D &state, const TTensor1D mean ){
			state = sample_binary( mean );
		}
		virtual void sample  ( STTensor2D &state, const STTensor2D &mean ){
			for( int i = 0; i < mean.y_max; ++ i ){
					STTensor1D line_state = state[ i ];
					STTensor1D line_mean = mean[ i ]; 
					sample ( line_state, line_mean );
			}
		}
        virtual void cal_mean( STTensor1D &mean, const STTensor1D &energy){
            mean = sigmoid( energy );
        }        
		virtual void cal_mean( STTensor2D &mean, const STTensor2D &energy ){
			for( int i = 0; i < mean.y_max; ++ i ){
					STTensor1D line_mean = mean[ i ];
					STTensor1D line_energy = energy[ i ]; 
					cal_mean ( line_mean, line_energy );
			}
		}
    };

    // gaussian node
    class CFSRBMGaussianNode : public CFSRBMNode{
    private:
        float sigma, sigma_sqr;
    public:
        CFSRBMGaussianNode( float sigma ){
            this->sigma     = sigma;
            this->sigma_sqr = sigma * sigma;
        }
        virtual void sample  ( STTensor1D &state, const STTensor1D &mean ){
            tensor::sample_gaussian( state, mean, sigma );
        }
		virtual void sample  ( STTensor2D &state, const STTensor2D &mean ){
			for( int i = 0; i < mean.y_max; ++ i ){
					STTensor1D line_state = state[ i ];
					STTensor1D line_mean = mean[ i ]; 
					sample ( line_state, line_mean );
			}
		}
        virtual void cal_mean( STTensor1D &mean , const STTensor1D &energy){
            mean = energy * sigma_sqr; 
        } 
		virtual void cal_mean( STTensor2D &mean, const STTensor2D &energy ){
			for( int i = 0; i < mean.y_max; ++ i ){
					STTensor1D line_mean = mean[ i ];
					STTensor1D line_energy = energy[ i ]; 
					cal_mean ( line_mean, line_energy );
			}
		}
    };

    inline CFSRBMNode *create_visible_node( const CFSRBMModelParam &param ){
        switch( param.model_type ){
        case 0: return new CFSRBMBinaryNode();
        case 1: return new CFSRBMGaussianNode( param.v_sigma );
        default: return NULL;
        }
    }
    inline CFSRBMNode *create_hidden_node( const CFSRBMModelParam &param ){
        switch( param.model_type ){
        case 0:
        case 1: return new CFSRBMBinaryNode();
        default: return NULL;
        }
    }

    // one layer of SRBM
    struct CFSRBMLayer{
        TTensor1D	h_bias;
		STTensor2D	v_bias;
        TTensor3D	W;
        CFSRBMNode *v_node, *h_node;
        STTensor2D	v_state;
        CFSRBMLayer(){}
        CFSRBMLayer( const CFSRBMModel &model ){
            W      = clone( model.Wvh );
            h_bias = clone( model.h_bias );
            v_bias = clone( model.v_bias );
			v_state.set_param( model.param.v_max);
            tensor::alloc_space( v_state );
            v_node = create_visible_node( model.param );
            h_node = create_hidden_node( model.param );

        }

        inline void free_space(){
            tensor::free_space( h_bias );
            tensor::free_space( v_bias );
            tensor::free_space( W );
            tensor::free_space( v_state );
            delete v_node; delete h_node;
        }
        
        // feed forward from v to h 
        inline void feed_forward( TTensor1D &h_state ){
			for( int i = 0; i < v_state.y_max; ++ i)
            	h_state += dot( v_state[ i ] , W[ i ] );
            h_state+= h_bias;
            h_node->cal_mean( h_state, h_state );
        }        
    };    

    // simple implementation of srbm
    class CFSRBMSimple:public CFSRBM{
    private:
        int  cd_step, sample_counter;
        CFSRBMTrainParam param;
        bool persistent_ok;
    private:
        TTensor1D 	d_h_bias;
		TTensor2D	d_v_bias;
        TTensor3D 	d_W;
    private:
        // nodes of each layer
        vector<SRBMLayer> layers;
        TTensor1D 	h_neg, h_pos;		// gavinhu: added l_neg.
		STTensor2D	v_neg;
    private:
        // initalize the space
        inline void init( const CFDBNModel &model ){
            for( size_t i = 0 ; i < model.layers.size() ; i ++ )
                layers.push_back( CFRBMLayer(model.layers[i]) );

            d_h_bias = clone( model.layers.back().d_h_bias );
            d_v_bias = clone( model.layers.back().d_v_bias );
            d_W      = clone( model.layers.back().d_Wvh );
            v_neg    = alloc_like( model.layers.back().d_v_bias );
            h_neg    = alloc_like( model.layers.back().d_h_bias );
            h_pos    = alloc_like( model.layers.back().d_h_bias );
            sample_counter = 0; persistent_ok = false;            
        }
    public:
        CFRBMSimple( const CFDBNModel &model, const CFRBMTrainParam &param ){
            init( model );
            this->param = param;
            // intialize the tensor engine
            init_tensor_engine( 0 );
        }
        
        // deallocate the space
        virtual ~SRBMSimple(){
            //destroy_tensor_engine();
            for( size_t i = 0 ; i < layers.size() ; i ++ )
                layers[i].free_space();
            layers.clear();
            tensor::free_space( d_h_bias );
            tensor::free_space( d_v_bias );
            tensor::free_space( d_W );
            tensor::free_space( v_neg );
            tensor::free_space( h_pos );
            tensor::free_space( h_neg );

            // destroy the tensor engine
            destroy_tensor_engine();
        }              
    private:

		inline void upfeed_soft_max( STTensor2D &soft_max, TTensor1D &h_pos, TTensor3D &W, TTensor1D &h_bias ){
			for(int i = 0; i < soft_max.y_max; ++ i){
					STTensor1D line = soft_max[ i ];
					h_pos += dot( line, W[ i ] );
			}
			h_pos += h_bias;
		}

		inline void downfeed_soft_max( TTensor1D &h_neg, TTensor2D &soft_max, TTensor3D &W, TTensor2D &v_bias ){
			for(int i = 0; i < soft_max.y_max; ++ i){
					TTensor1D line = soft_max[ i ];
					line = dot( h_neg, W[ i ].T() );
					line += v_bias[ i ]; 
			}
		}

		inline void normalize( TTensor2D &soft_max ){
				TENSOR_FLOAT sum[ soft_max.x_max ];
				memset( sum, 0, sum.size() );
				for( int i = 0; i < soft_max.y_max; ++ i){
						TTensor1D line = soft_max[ i ];
						for( int j = 0; j < line.x_max; ++ j )
								sum[ j ] += line[ j ];
				}
				for( int i = 0; i < soft_max.y_max; ++ i){
						TTensor1D line = soft_max[ i ];
						for( int j = 0; j < line.x_max; ++ j )
								line[ j ] /= sum[ j ];
				}
		}
        // calculate the datas in cd steps	
        inline void cal_cd_steps( STTensor2D &v_pos, STTensor2D &v_neg,
                                  TTensor1D &h_pos, TTensor1D &h_neg,
                                  TTensor1D &h_persistent ){
            TTensor1D & h_bias = layers.back().h_bias;
            TTensor2D & v_bias = layers.back().v_bias;
            TTensor3D & W      = layers.back().W;
            ISRBMNode *h_node  = layers.back().h_node;
            ISRBMNode *v_node  = layers.back().v_node;

            // go up
			upfeed_soft_max( v_pos, h_pos, W, h_bias ); 
            h_node->cal_mean( h_pos, h_pos );
            // negative steps
            for( int i = 0 ; i < cd_step ; i ++ ){
                TTensor1D &hh = ( i == 0 ? h_persistent : h_neg );
                // sample h
                h_node->sample( h_neg, hh );

                // go down
				downfeed_soft_max( h_neg, v_neg, W, v_bias );
				normalize( v_neg );
                v_node->cal_mean( v_neg, v_neg );
                v_node->sample  ( v_neg, v_neg );

                // go up
				upfeed_soft_max( v_neg, h_neg, W, h_bias );
                h_node->cal_mean( h_neg, h_neg );
            }
        }
	        // update the weight of the last layer
        inline void update_weight(){
            TTensor1D & h_bias = layers.back().h_bias;
            TTensor2D & v_bias = layers.back().v_bias;
            TTensor3D & W      = layers.back().W;

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
            TTensor2D &v_pos = layers.back().v_state;

            // whether can be use persistent chain
            TTensor1D &hp = persistent_ok ? h_neg : h_pos;

			// gavinhu: added l_pos, l_neg
            cal_cd_steps( v_pos, v_neg,  h_pos, h_neg, hp );

            persistent_ok = ( param.persistent_cd !=0 );

            // calculate the gradient
			for( int i = 0; i < d_W.z_max; ++ i ){
				TTensor2D line = d_W[ i ];
            	line += dot( v_pos.T(), h_pos );
            	line -= dot( v_neg.T(), h_neg );
			}

            if( param.chg_hidden_bias ){
                d_h_bias += h_pos;
                d_h_bias -= h_neg; 
            }
            if( param.chg_visible_bias ){
                d_v_bias += v_pos;
                d_v_bias -= v_neg;
            }

            if( ++sample_counter == param.batch_size ){
                update_weight();
                sample_counter = 0;
            }
        }
        
        inline void setup_input( const apex_tensor::CSTensor2D &data ){
            tensor::copy( layers[0].v_state , data );
            for( size_t i = 1 ; i < layers.size() ; i ++ ){
                layers[i-1].feed_forward( layers[i].v_state );
            }  
        }

    public:
        virtual void train_update( const apex_tensor::CSTensor2D &data ){
            setup_input( data );
            train_update();
        }
        virtual void train_update_trunk( const apex_tensor::CSTensor3D &data ){
            for( int i = 0 ; i < data.y_max ; i ++ )
                train_update( data[i] );
        }

        // do validation, return the statistics
        virtual void validate_stats( CFRBMModelStats &stats, const apex_tensor::CSTensor3D &data ){
            TTensor3D	grad_W;
            TTensor1D	pos_grad_h, neg_grad_h, loss;
		   	STTensor2D	pos_grad_v, neg_grad_v;
            grad_W = clone( stats.grad_W );
            pos_grad_h = clone( stats.pos_grad_h );
            neg_grad_h = clone( stats.neg_grad_h );
            pos_grad_v = clone( stats.pos_grad_v );
            neg_grad_v = clone( stats.neg_grad_v );
            loss       = clone( stats.loss );

            for( int i = 0 ; i < data.y_max ; i ++ ){
                setup_input( data[i] );

                STTensor2D &v_pos = layers.back().v_state;
                // whether can be use persistent chain
                TTensor1D &hp = persistent_ok ? h_neg : h_pos;
                persistent_ok = ( param.persistent_cd !=0 );
				
               	for( int i = 0; i < grad_W.z_max; ++ i ){
					TTensor2D line = grad_W[ i ];
            		line += dot( v_pos.T(), h_pos );
            		line -= dot( v_neg.T(), h_neg );
				}
                
                pos_grad_h += h_pos;
                neg_grad_h -= h_neg;                                 
                pos_grad_v += v_pos;
                neg_grad_v -= v_neg;
                v_neg      -= v_pos;
                v_neg       = v_neg * v_neg;
                loss       += v_neg;
            }                       
            stats.sample_counter += (int)data.y_max;
            apex_tensor::tensor::copy( stats.grad_W, grad_W );
            apex_tensor::tensor::copy( stats.pos_grad_h, pos_grad_h );
            apex_tensor::tensor::copy( stats.neg_grad_h, neg_grad_h );
            apex_tensor::tensor::copy( stats.pos_grad_v, pos_grad_v );
            apex_tensor::tensor::copy( stats.neg_grad_v, neg_grad_v );
            apex_tensor::tensor::copy( stats.loss, loss );
            
            apex_tensor::tensor::free_space( grad_W );
            apex_tensor::tensor::free_space( pos_grad_h );
            apex_tensor::tensor::free_space( neg_grad_h );
            apex_tensor::tensor::free_space( pos_grad_v );
            apex_tensor::tensor::free_space( neg_grad_v );
            apex_tensor::tensor::free_space( loss );
        }

        /* clone model trainied to model */
        virtual void clone_model( CFDBNModel &model )const{
            if( model.layers.size() != layers.size() ){
                printf("error model size\n"); exit( -1 );
            } 

            CFRBMModel &md = model.layers.back();
            const CFRBMLayer &mm = layers.back();
            
            apex_tensor::tensor::copy( md.Wvh , mm.W );
            apex_tensor::tensor::copy( md.h_bias , mm.h_bias );
            apex_tensor::tensor::copy( md.v_bias , mm.v_bias );
            apex_tensor::tensor::copy( md.d_Wvh , d_W );
            apex_tensor::tensor::copy( md.d_h_bias , d_h_bias );
            apex_tensor::tensor::copy( md.d_v_bias , d_v_bias );   
			
		}       

        /* set steps of CD */
        virtual void set_cd_step( int cd_step ){
            this->cd_step = cd_step;
        }                
    };
    
    namespace factory{
        // create a stacked rbm
        CFRBM *create_cfrbm( const CFDBNModel &model, const CFRBMTrainParam &param ){
            return new CFSRBMSimple( model, param );
        }
    };
};

#endif

