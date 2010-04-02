#ifndef _APEX_SRBM_CPP_
#define _APEX_SRBM_CPP_

#include "apex_srbm.h"
#include "apex_srbm_model.h"
#include "apex_srbm_model_stats.h"
#include "../tensor/apex_tensor.h"

#include <vector>

namespace apex_rbm{
    using namespace std;
    using namespace apex_tensor;

    //node of SRBM
    class ISRBMNode{
    public:
        virtual void sample  ( TTensor1D &state, const TTensor1D &mean ) = 0;
        virtual void cal_mean( TTensor1D &mean , const TTensor1D &energy) = 0;
    };
    // bianry node
    class SRBMBinaryNode : public ISRBMNode{
    public:
        virtual void sample  ( TTensor1D &state, const TTensor1D &mean ){
            state = sample_binary( mean );
        }
        virtual void cal_mean( TTensor1D &mean , const TTensor1D &energy){
            mean = sigmoid( energy );
        }        
    };

    // gaussian node
    class SRBMGaussianNode : public ISRBMNode{
    private:
        float sigma, sigma_sqr;
    public:
        SRBMGaussianNode( float sigma ){
            this->sigma     = sigma;
            this->sigma_sqr = sigma * sigma;
        } 
        virtual void sample  ( TTensor1D &state, const TTensor1D &mean ){
            tensor::sample_gaussian( state, mean, sigma );
        }
        virtual void cal_mean( TTensor1D &mean , const TTensor1D &energy){
            mean = energy * sigma_sqr; 
        }        
    };

    inline ISRBMNode *create_visible_node( const SRBMModelParam &param ){
        switch( param.model_type ){
        case 0: return new SRBMBinaryNode();
        case 1: return new SRBMGaussianNode( param.v_sigma );
        default: return NULL;
        }
    }
    inline ISRBMNode *create_hidden_node( const SRBMModelParam &param ){
        switch( param.model_type ){
        case 0:
        case 1: return new SRBMBinaryNode();
        default: return NULL;
        }
    }

	// gavinhu: add support for labeled rbm.
	// * currently we only support binary labels
	inline ISRBMNode *create_label_node(const SRBMModelParam &param) {
		switch (param.model_type) {
		case 0:
		case 1: return new SRBMBinaryNode();
		default: return NULL;
		}
	}

    
    // one layer of SRBM
    struct SRBMLayer{
        TTensor1D h_bias  , v_bias, l_bias;			// gavinhu: l_bias for label bias.
        TTensor2D W, Wlh;							// gavinhu: Wlh for connections between labels and hidden units.
        ISRBMNode *v_node, *h_node, *l_node;		// gavinhu: l_node for label node.
        TTensor1D v_state, l_state;					// gavinhu: l_state for label states.
        SRBMLayer(){}
        SRBMLayer( const SRBMModel &model ){
            W      = clone( model.Wvh );
            h_bias = clone( model.h_bias );
            v_bias = clone( model.v_bias );
			v_state.set_param( model.param.v_max);
            tensor::alloc_space( v_state );

			// gavinhu
			if (model.param.l_max > 0) {
				l_bias = clone(model.l_bias);
				Wlh = clone(model.Wlh);
				l_state.set_param(model.param.l_max);
				tensor::alloc_space(l_state);
				l_node = create_label_node(model.param);
			} else {
				l_node = NULL;
			}

            v_node = create_visible_node( model.param );
            h_node = create_hidden_node( model.param );

        }

        inline void free_space(){
            tensor::free_space( h_bias );
            tensor::free_space( v_bias );
            tensor::free_space( W );
            tensor::free_space( v_state );
            delete v_node; delete h_node;

			// gavinhu
			if (l_node != NULL) {
				tensor::free_space(l_bias);
				tensor::free_space(l_state);
				tensor::free_space(Wlh);
				delete l_node;
			}
        }
        
        // feed forward from v to h 
        inline void feed_forward( TTensor1D &h_state ){
            h_state = dot( v_state , W );

			// gavinhu
			if (l_node != NULL)
				h_state += dot(l_state, Wlh);

            h_state+= h_bias;
            h_node->cal_mean( h_state, h_state );
        }        
    };    

    // simple implementation of srbm
    class SRBMSimple:public ISRBM{
    private:
        int  cd_step, sample_counter;
        SRBMTrainParam param;
        bool persistent_ok;
    private:
        TTensor1D d_h_bias, d_v_bias;
        TTensor2D d_W;

		TTensor1D d_l_bias;			// gavinhu
		TTensor2D d_Wlh;			// gavinhu

    private:
        // nodes of each layer
        vector<SRBMLayer> layers;
        TTensor1D v_neg,  h_neg, h_pos, l_neg;		// gavinhu: added l_neg.
    private:
        // initalize the space
        inline void init( const SDBNModel &model ){
            for( size_t i = 0 ; i < model.layers.size() ; i ++ )
                layers.push_back( SRBMLayer(model.layers[i]) );

            d_h_bias = clone( model.layers.back().d_h_bias );
            d_v_bias = clone( model.layers.back().d_v_bias );
            d_W      = clone( model.layers.back().d_Wvh );
            v_neg    = alloc_like( model.layers.back().d_v_bias );
            h_neg    = alloc_like( model.layers.back().d_h_bias );
            h_pos    = alloc_like( model.layers.back().d_h_bias );

			// gavinhu
			if (layers.back().l_node != NULL) {
				d_l_bias = clone(model.layers.back().d_l_bias);
				d_Wlh = clone(model.layers.back().d_Wlh);
				l_neg = alloc_like(model.layers.back().d_l_bias);
			}
            
            sample_counter = 0; persistent_ok = false;            
        }
    public:
        SRBMSimple( const SDBNModel &model, const SRBMTrainParam &param ){
            init( model );
            this->param = param;
            // intialize the tensor engine
            init_tensor_engine( 0 );
        }
        
        // deallocate the space
        virtual ~SRBMSimple(){
            //destroy_tensor_engine();

			// gavinhu
			if (layers.back().l_node != NULL) {
				tensor::free_space(d_l_bias);
				tensor::free_space(d_Wlh);
				tensor::free_space(l_neg);
			}

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
        // calculate the datas in cd steps
        inline void cal_cd_steps( TTensor1D &l_pos, TTensor1D &l_neg,
                                  TTensor1D &v_pos, TTensor1D &v_neg,
                                  TTensor1D &h_pos, TTensor1D &h_neg,
                                  TTensor1D &h_persistent ){
            TTensor1D & h_bias = layers.back().h_bias;
            TTensor1D & v_bias = layers.back().v_bias;
			TTensor1D & l_bias = layers.back().l_bias;		// gavinhu
            TTensor2D & W      = layers.back().W;
			TTensor2D & Wlh    = layers.back().Wlh;			// gavinhu
            ISRBMNode *h_node  = layers.back().h_node;
            ISRBMNode *v_node  = layers.back().v_node;
			ISRBMNode *l_node  = layers.back().l_node;		// gavinhu

            // go up
            h_pos   = dot( v_pos, W );
            h_pos  += h_bias;

			// gavinhu
			if (l_node != NULL)
				h_pos += dot(l_pos, Wlh);

            h_node->cal_mean( h_pos, h_pos );
            // negative steps
            for( int i = 0 ; i < cd_step ; i ++ ){
                TTensor1D &hh = ( i == 0 ? h_persistent : h_neg );
                // sample h
                h_node->sample( h_neg, hh );

                // go down
                v_neg = dot( h_neg, W.T() );
                v_neg+= v_bias;
                v_node->cal_mean( v_neg, v_neg );
                v_node->sample  ( v_neg, v_neg );

				// gavinhu: go down for labels
				if (l_node != NULL) {
					l_neg = dot(h_neg, Wlh.T());
					l_neg += l_bias;
					l_node->cal_mean(l_neg, l_neg);
					l_node->sample(l_neg, l_neg);
				}

                // go up
                h_neg = dot( v_neg,  W );
                h_neg+= h_bias;

				// gavinhu
				if (l_node != NULL)
					h_neg += dot(l_neg, Wlh);

                h_node->cal_mean( h_neg, h_neg );
            }                                    
        }

        // update the weight of the last layer
        inline void update_weight(){
            TTensor1D & h_bias = layers.back().h_bias;
            TTensor1D & v_bias = layers.back().v_bias;
            TTensor2D & W      = layers.back().W;
			TTensor1D & l_bias = layers.back().l_bias;	// gavinhu
			TTensor2D & Wlh    = layers.back().Wlh;		// gavinhu

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

			// gavinhu
			if (layers.back().l_node != NULL) {
				Wlh = Wlh * (1-eta*param.wd_W) + d_Wlh * eta;
				d_Wlh *= param.momentum;

				if (param.chg_visible_bias) {			// FIXME: maybe chg_label_bias??
					l_bias = l_bias * (1-eta*param.wd_v) + d_l_bias * eta;
					d_l_bias *= param.momentum;
				}
			}
		}

        // update in training
        inline void train_update(){
            TTensor1D &v_pos = layers.back().v_state;
			TTensor1D &l_pos = layers.back().l_state;	// gavinhu

            // whether can be use persistent chain
            TTensor1D &hp = persistent_ok ? h_neg : h_pos;

			// gavinhu: added l_pos, l_neg
            cal_cd_steps( l_pos, l_neg, v_pos, v_neg,  h_pos, h_neg, hp );

            persistent_ok = ( param.persistent_cd !=0 );

            // calculate the gradient
            d_W += dot( v_pos.T(), h_pos );
            d_W -= dot( v_neg.T(), h_neg );

			// gavinhu
			if (layers.back().l_node != NULL) {
				d_Wlh += dot(l_pos.T(), h_pos);
				d_Wlh -= dot(l_neg.T(), h_neg);
			}

            if( param.chg_hidden_bias ){
                d_h_bias += h_pos;
                d_h_bias -= h_neg; 
            }
            if( param.chg_visible_bias ){
                d_v_bias += v_pos;
                d_v_bias -= v_neg;
            }

			// gavinhu
			if (param.chg_visible_bias) {		// FIXME: maybe chg_label_bias??
				d_l_bias += l_pos;
				d_l_bias -= l_neg;
			}
            
            if( ++sample_counter == param.batch_size ){
                update_weight();
                sample_counter = 0;
            }
        }
        
        inline void setup_input( const apex_tensor::CTensor1D &data ){
            tensor::copy( layers[0].v_state , data );
            for( size_t i = 1 ; i < layers.size() ; i ++ ){
                layers[i-1].feed_forward( layers[i].v_state );
            }  
        }

		// gavinhu
		inline void setup_input(const apex_tensor::CTensor1D &label, const apex_tensor::CTensor1D &data) {
			if (layers[0].l_node != NULL)
				tensor::copy(layers[0].l_state, label);

			tensor::copy(layers[0].v_state, data);
			for (size_t i = 1; i < layers.size(); i++) {
				if (layers[i].l_node != NULL)
					tensor::copy(layers[i].l_state, label);		// simply copy the label to every labeled rbm.

				layers[i-1].feed_forward(layers[i].v_state);
			}
		}

    public:
        virtual void train_update( const apex_tensor::CTensor1D &data ){
            setup_input( data );
            train_update();
        }
        virtual void train_update_trunk( const apex_tensor::CTensor2D &data ){
            for( int i = 0 ; i < data.y_max ; i ++ )
                train_update( data[i] );
        }

		// gavinhu
		virtual void train_update(const apex_tensor::CTensor1D &label, const apex_tensor::CTensor1D &data) {
			setup_input(label, data);
			train_update();
		}

		// gavinhu
		virtual void train_update_trunk(const apex_tensor::CTensor2D &label, const apex_tensor::CTensor2D &data) {
			if (label.y_max != data.y_max) {
				printf("Warning: label size does not match the dataset.\n");
			}
			for (int i = 0; i < label.y_max && i < data.y_max; i++)
				train_update(label[i], data[i]);
		}

        // do validation, return the statistics
        virtual void validate_stats( SRBMModelStats &stats, const apex_tensor::CTensor2D &data ){
            TTensor2D grad_W;
            TTensor1D pos_grad_h, neg_grad_h, pos_grad_v, neg_grad_v, loss;
            grad_W = clone( stats.grad_W );
            pos_grad_h = clone( stats.pos_grad_h );
            neg_grad_h = clone( stats.neg_grad_h );
            pos_grad_v = clone( stats.pos_grad_v );
            neg_grad_v = clone( stats.neg_grad_v );
            loss       = clone( stats.loss );

            for( int i = 0 ; i < data.y_max ; i ++ ){
                setup_input( data[i] );

                TTensor1D &v_pos = layers.back().v_state;
				TTensor1D &l_pos = layers.back().l_state;		// gavinhu
                // whether can be use persistent chain
                TTensor1D &hp = persistent_ok ? h_neg : h_pos;
				cal_cd_steps( l_pos, l_neg, v_pos, v_neg, h_pos, h_neg, hp );		// gavinhu
                persistent_ok = ( param.persistent_cd !=0 );
                
                grad_W += dot( v_pos.T() , h_pos );
                grad_W -= dot( v_neg.T() , h_neg );
                
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

		// gavinhu
		virtual void validate_stats(apex_rbm::SRBMModelStats &stats, const apex_tensor::CTensor2D &label, const apex_tensor::CTensor2D &data) {
			if (label.y_max != data.y_max) {
				printf("Warning: label size does not match the dataset.\n");
			}

            TTensor2D grad_W;
            TTensor1D pos_grad_h, neg_grad_h, pos_grad_v, neg_grad_v, loss;
            grad_W = clone( stats.grad_W );
            pos_grad_h = clone( stats.pos_grad_h );
            neg_grad_h = clone( stats.neg_grad_h );
            pos_grad_v = clone( stats.pos_grad_v );
            neg_grad_v = clone( stats.neg_grad_v );
            loss       = clone( stats.loss );

			// gavinhu
			TTensor2D grad_Wlh;
			TTensor1D pos_grad_l, neg_grad_l, lloss;
			if (layers.back().l_node != NULL) {
				grad_Wlh   = clone(stats.grad_Wlh);
				pos_grad_l = clone(stats.pos_grad_l);
				neg_grad_l = clone(stats.neg_grad_l);
				lloss      = clone(stats.lloss);
			}

			for( int i = 0 ; i < label.y_max && i < data.y_max ; i ++ ){

                setup_input( label[i], data[i] );

                TTensor1D &v_pos = layers.back().v_state;                
				TTensor1D &l_pos = layers.back().l_state;		// gavinhu
                // whether can be use persistent chain
                TTensor1D &hp = persistent_ok ? h_neg : h_pos;
                cal_cd_steps( l_pos, l_neg, v_pos, v_neg, h_pos, h_neg, hp );		// gavinhu
                persistent_ok = ( param.persistent_cd !=0 );
                
                grad_W += dot( v_pos.T() , h_pos );
                grad_W -= dot( v_neg.T() , h_neg );

				pos_grad_h += h_pos;
                neg_grad_h -= h_neg;                                 
                pos_grad_v += v_pos;
                neg_grad_v -= v_neg;
                v_neg      -= v_pos;
                v_neg       = v_neg * v_neg;
                loss       += v_neg;

				// gavinhu
				if (layers.back().l_node != NULL) {
					grad_Wlh += dot(l_pos.T(), h_pos);
					grad_Wlh -= dot(l_neg.T(), h_neg);
					pos_grad_l += l_pos;
					neg_grad_l -= l_neg;
					l_neg      -= l_pos;
					l_neg       = l_neg * l_neg;
					lloss      += l_neg;
				}
            }                       
            stats.sample_counter += (int)data.y_max;
            apex_tensor::tensor::copy( stats.grad_W, grad_W );
            apex_tensor::tensor::copy( stats.pos_grad_h, pos_grad_h );
            apex_tensor::tensor::copy( stats.neg_grad_h, neg_grad_h );
            apex_tensor::tensor::copy( stats.pos_grad_v, pos_grad_v );
            apex_tensor::tensor::copy( stats.neg_grad_v, neg_grad_v );
            apex_tensor::tensor::copy( stats.loss, loss );

			// gavinhu
			if (layers.back().l_node != NULL) {
				apex_tensor::tensor::copy(stats.grad_Wlh, grad_Wlh);
				apex_tensor::tensor::copy(stats.pos_grad_l, pos_grad_l);
				apex_tensor::tensor::copy(stats.neg_grad_l, neg_grad_l);
				apex_tensor::tensor::copy(stats.lloss, lloss);

				apex_tensor::tensor::free_space(grad_Wlh);
				apex_tensor::tensor::free_space(pos_grad_l);
				apex_tensor::tensor::free_space(neg_grad_l);
				apex_tensor::tensor::free_space(lloss);
			}
            
            apex_tensor::tensor::free_space( grad_W );
            apex_tensor::tensor::free_space( pos_grad_h );
            apex_tensor::tensor::free_space( neg_grad_h );
            apex_tensor::tensor::free_space( pos_grad_v );
            apex_tensor::tensor::free_space( neg_grad_v );
            apex_tensor::tensor::free_space( loss );
		}
       
        /* clone model trainied to model */
        virtual void clone_model( SDBNModel &model )const{
            if( model.layers.size() != layers.size() ){
                printf("error model size\n"); exit( -1 );
            } 

            SRBMModel &md = model.layers.back();
            const SRBMLayer &mm = layers.back();
            
            apex_tensor::tensor::copy( md.Wvh , mm.W );
            apex_tensor::tensor::copy( md.h_bias , mm.h_bias );
            apex_tensor::tensor::copy( md.v_bias , mm.v_bias );
            apex_tensor::tensor::copy( md.d_Wvh , d_W );
            apex_tensor::tensor::copy( md.d_h_bias , d_h_bias );
            apex_tensor::tensor::copy( md.d_v_bias , d_v_bias );   
			
			// gavinhu
			if (md.param.l_max > 0) {
				apex_tensor::tensor::copy(md.Wlh, mm.Wlh);
				apex_tensor::tensor::copy(md.l_bias, mm.l_bias);
				apex_tensor::tensor::copy(md.d_Wlh, d_Wlh);
				apex_tensor::tensor::copy(md.d_l_bias, d_l_bias);
			}
        }       

        /* set steps of CD */
        virtual void set_cd_step( int cd_step ){
            this->cd_step = cd_step;
        }                
    };
    
    namespace factory{
        // create a stacked rbm
        ISRBM *create_srbm( const SDBNModel &model, const SRBMTrainParam &param ){
            return new SRBMSimple( model, param );
        }
    };
};

#endif

