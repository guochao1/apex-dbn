#ifndef _APEX_CRBM_CPP_
#define _APEX_CRBM_CPP_

#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "apex_crbm.h"
#include "apex_crbm_model.h"
#include "apex_crbm_model_stats.h"
#include "../tensor/apex_tensor.h"

#include <vector>

namespace apex_rbm{
    using namespace std;
    using namespace apex_tensor;    
    //node of SRBM
    class ICRBMNode{
    public:
        virtual void sample  ( TTensor3D &state, const TTensor3D &mean )const = 0;
        virtual void cal_mean( TTensor3D &mean , const TTensor3D &energy)const = 0;
        // feed forward data needed   
        virtual void feed_forward( TTensor3D &v_next, const TTensor3D &h_curr )const = 0;
        // feed forward bias to next layer 
        virtual void forward_bias( TTensor1D &v_bias_next, const TTensor1D &h_bias_curr )const = 0;
        // reget the bound of data
        virtual void reget_bound ( int &input_y_max, int &input_x_max  )const = 0;
        // reget the bound of hidden data 
        virtual void reget_hidden_bound( int &h_y_max, int &h_x_max )const = 0;
        // calculate sparse_regularization
        virtual void sparse_reg( TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad, const TTensor3D &h_pos )const = 0;
   };

    // bianry node
    class CRBMBinaryNode : public ICRBMNode{
    public:
        virtual void sample  ( TTensor3D &state, const TTensor3D &mean )const{
            state = sample_binary( mean );
        }
        virtual void cal_mean( TTensor3D &mean , const TTensor3D &energy)const{
            mean = sigmoid( energy );
        }               
        virtual void feed_forward( TTensor3D &v_next, const TTensor3D &h_curr )const{
            tensor::crbm::copy_fit( v_next, h_curr );
        }
        
        virtual void forward_bias( TTensor1D &v_bias_next, const TTensor1D &h_bias_curr )const{
            tensor::copy( v_bias_next, h_bias_curr );
        }
        // reget the bound of data
        virtual void reget_bound ( int &input_y_max, int &input_x_max  )const{}
        
        // reget the bound of hidden data 
        virtual void reget_hidden_bound( int &h_y_max, int &h_x_max )const{ }
        virtual void sparse_reg( TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad, const TTensor3D &h_pos )const{
            tensor::crbm::add_sparse_info( h_sum_mf, h_sum_mf_grad, h_pos , 1 );
        }
    };

    template<bool use_type2>
    class CRBMGaussianNode : public ICRBMNode{
    private:
        float sigma, sigma_sqr;
    public:
        CRBMGaussianNode( float sigma ){
            this->sigma       = sigma;
            this->sigma_sqr   = sigma*sigma;
        }
        virtual void sample  ( TTensor3D &state, const TTensor3D &mean )const{
            state = sample_gaussian( mean, sigma );
        }
        virtual void cal_mean( TTensor3D &mean , const TTensor3D &energy )const{
            if( use_type2 ){
                if( mean.elem != energy.elem ) tensor::copy( mean, energy );
            }else{
                mean =  energy * sigma_sqr;
            }
        }               
        virtual void feed_forward( TTensor3D &v_next, const TTensor3D &h_curr )const{
            tensor::crbm::copy_fit( v_next, h_curr );
        }
        
        virtual void forward_bias( TTensor1D &v_bias_next, const TTensor1D &h_bias_curr )const{
            tensor::copy( v_bias_next, h_bias_curr );
        }
        // reget the bound of data
        virtual void reget_bound ( int &input_y_max, int &input_x_max  )const{}        
        // reget the bound of hidden data 
        virtual void reget_hidden_bound( int &h_y_max, int &h_x_max )const{}
        virtual void sparse_reg( TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad, const TTensor3D &h_pos )const{
            tensor::crbm::add_sparse_info( h_sum_mf, h_sum_mf_grad, h_pos , 1 );
        }
    };
    

    // maxpooling node 
    template<bool scale_energy>
    class CRBMMaxpoolNode : public ICRBMNode{
    private:
        int pool_size;
        float energy_scale;
    public:
        CRBMMaxpoolNode( const CRBMModelParam &param, float energy_scale = 1.0f ){
            pool_size = param.pool_size;
            this->energy_scale = energy_scale;
        }
        virtual void sample  ( TTensor3D &state, const TTensor3D &mean )const{
            tensor::crbm::sample_maxpooling_2D( state, mean, pool_size );
        }
        virtual void cal_mean( TTensor3D &mean , const TTensor3D &energy )const{
            if( !scale_energy ){ 
                tensor::crbm::norm_maxpooling_2D( mean, energy, pool_size );
            }else{
                mean = energy * energy_scale;
                tensor::crbm::norm_maxpooling_2D( mean, mean, pool_size );
            }
        }   
        
        virtual void forward_bias( TTensor1D &v_bias_next, const TTensor1D &h_bias_curr )const{
			tensor::copy( v_bias_next, h_bias_curr);
			v_bias_next += (float)( 2.0 * log((double)pool_size) );
        }
        virtual void feed_forward( TTensor3D &v_next, const TTensor3D &h_curr )const{
            tensor::crbm::pool_up( v_next, h_curr, pool_size );
        }
        // reget the bound of data
        virtual void reget_bound ( int &input_y_max, int &input_x_max  )const{
            input_x_max /= pool_size;
            input_y_max /= pool_size;
        }
        // reget the bound of hidden data 
        virtual void reget_hidden_bound( int &h_y_max, int &h_x_max )const{
            h_y_max = (h_y_max / pool_size) * pool_size;
            h_x_max = (h_x_max / pool_size) * pool_size;
        }
        virtual void sparse_reg( TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad, const TTensor3D &h_pos )const{
            tensor::crbm::add_sparse_info( h_sum_mf, h_sum_mf_grad, h_pos , pool_size );
        }
    };


    inline ICRBMNode *create_visible_node( const CRBMModelParam &param ){
        switch( param.model_type ){
        case 0: return new CRBMBinaryNode();
        case 1: return new CRBMGaussianNode<false>( param.v_sigma );
        case 2: return new CRBMGaussianNode<true> ( param.v_sigma );
        default: return NULL;
        }
    }
    inline ICRBMNode *create_hidden_node( const CRBMModelParam &param ){
        switch( param.model_type ){
        case 0: return new CRBMMaxpoolNode<false>( param );
        case 1: return new CRBMMaxpoolNode<false>( param );
        case 2: return new CRBMMaxpoolNode<true> ( param, 1.0f/(param.v_sigma*param.v_sigma) );
        default: return NULL;
        }
    }
    
    // one layer of CRBM
    struct CRBMLayer{
        TTensor1D h_bias  , v_bias;
        TTensor4D W;
        ICRBMNode *v_node, *h_node;        
        TTensor3D v_state, h_state;
        CRBMLayer(){}
        CRBMLayer( const CRBMModel &model, int y_max, int x_max ){
            v_node = create_visible_node( model.param );
            h_node = create_hidden_node ( model.param );

            W      = clone( model.W );
            h_bias = clone( model.h_bias );
            v_bias = clone( model.v_bias );          

            // we fit the input size to ensure perfect pooling
            int h_y_max = y_max - W.y_max + 1;
            int h_x_max = x_max - W.x_max + 1;
            h_node->reget_hidden_bound( h_y_max, h_x_max );

            v_state.set_param( model.param.v_max, h_y_max+W.y_max-1, h_x_max+W.x_max-1 );
            h_state.set_param( model.param.h_max, h_y_max, h_x_max );
            tensor::alloc_space( v_state );
            tensor::alloc_space( h_state );
        }

        inline void free_space(){
            tensor::free_space( h_bias );
            tensor::free_space( v_bias );
            tensor::free_space( W );
            tensor::free_space( v_state );
            tensor::free_space( h_state );
            delete v_node; delete h_node;
        }
        
        inline void forward_bias( TTensor1D &v_bias_next )const{
            h_node->forward_bias( v_bias_next, h_bias );
        }
        
        // feed forward from v to h 
        inline void feed_forward( TTensor3D &v_state_next ){
            tensor::crbm::conv2_r_valid( h_state, v_state, W, h_bias );
            h_node->cal_mean( h_state, h_state );
            h_node->feed_forward( v_state_next, h_state );
        }        
        
        inline void reget_bound( int &y_max, int &x_max ) const{
            y_max = y_max - W.y_max + 1;  
            x_max = x_max - W.x_max + 1;  
            h_node->reget_bound( y_max, x_max );
        }
        
        inline void sparse_reg( TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad ) const{
            h_node->sparse_reg( h_sum_mf, h_sum_mf_grad, h_state );
        }
    };   
 
    
    // simple implementation of srbm
    class CRBMSimple: public ICRBM{
    private:
        int  cd_step, h_size, v_size, vv_size;
        CRBMTrainParam param;
        bool persistent_ok;  
    private:
        int sample_counter, state_counter;
    private:
        TTensor1D d_h_bias, d_v_bias;
        TTensor1D h_sum_mf, h_sum_mf_grad;
        TTensor4D d_W;
        TTensor2D wd_sum;
    private:
        // nodes of each layer
        vector<CRBMLayer> layers;
        TTensor3D v_neg,  h_neg;
    private:
        // initalize the space
        inline void init( const CDBNModel &model, 
                          int input_y_max, int input_x_max ){
            for( size_t i = 0 ; i < model.layers.size() ; i ++ ){                
                layers.push_back( CRBMLayer(model.layers[i], input_y_max, input_x_max ) );
                layers.back().reget_bound( input_y_max, input_x_max );
            }

            d_h_bias = clone( model.layers.back().d_h_bias );
            d_v_bias = clone( model.layers.back().d_v_bias );
            d_W      = clone( model.layers.back().d_W );
  
            h_sum_mf      = alloc_like( d_h_bias );
            h_sum_mf_grad = alloc_like( d_h_bias );
            v_neg         = alloc_like( layers.back().v_state );
            h_neg         = alloc_like( layers.back().h_state );

            wd_sum.set_param( d_W.h_max, d_W.z_max );
            tensor::alloc_space( wd_sum );
            
            sample_counter = 0; state_counter = 0; persistent_ok = false;            
            
            h_sum_mf = 0.0f; h_sum_mf_grad = 0.0f;            
            h_size   = layers.back().h_state.y_max * layers.back().h_state.x_max;
            v_size   = layers.back().v_state.y_max * layers.back().v_state.x_max;

            if( param.refill_edge_area ){
                vv_size = (layers.back().h_state.y_max-d_W.y_max+1) * (layers.back().h_state.x_max-d_W.x_max+1);	
            }else{
                vv_size = v_size;
            }
            
            if( param.num_non_sparse_node > 0 ){
                TTensor4D & W      = layers.back().W;            
                W.z_max =  param.num_non_sparse_node;
                W = 0.0f;
                W.z_max = d_W.z_max;
            }
        }

        inline void init_async(){
            init_stream_engine( 3 );            
            // initialize the stream engine to support aynchronize speed up
            TTensor1D & h_bias = layers.back().h_bias;
            TTensor1D & v_bias = layers.back().v_bias;
            TTensor4D & W      = layers.back().W;            
            async::set_dependecy( d_v_bias, 1 );
            async::set_dependecy(   v_bias, 1 );
            async::set_dependecy( d_h_bias, 2 );
            async::set_dependecy(   h_bias, 2 );
            async::set_dependecy( h_sum_mf, 2 );
            async::set_dependecy( h_sum_mf_grad, 2 );
            async::set_dependecy( d_W     , 3 );
            async::set_dependecy(   W     , 3 );
            async::set_dependecy( wd_sum  , 3 );
        }
    public:
        CRBMSimple( const CDBNModel &model, const CRBMTrainParam &param ){
            // intialize the tensor engine
            init_tensor_engine( 0 );
			
			this->param = param;
            init( model, param.input_y_max, param.input_x_max );

            // forward_bias
            if( param.forward_bias!=0 && layers.size() > 1 ){
                layers[ layers.size()-2 ].forward_bias( layers.back().v_bias );
            }

            init_async();
            
            printf("CRBM initialized, h_size=%d, v_size=%d, vv_size=%d\n", h_size, v_size, vv_size );
        }
        
        // deallocate the space
        virtual ~CRBMSimple(){
            for( size_t i = 0 ; i < layers.size() ; i ++ )
                layers[i].free_space();
            layers.clear();
            tensor::free_space( d_h_bias );
            tensor::free_space( d_v_bias );
            tensor::free_space( d_W );
            tensor::free_space( wd_sum );
            tensor::free_space( h_sum_mf );
            tensor::free_space( h_sum_mf_grad );
            tensor::free_space( v_neg );
            tensor::free_space( h_neg );
            // destroy the tensor engine
            destroy_tensor_engine();
            destroy_stream_engine();
        }              
    private:
        // calculate the datas in cd steps
        inline void cal_cd_steps( TTensor3D &v_pos, TTensor3D &v_neg, 
                                  TTensor3D &h_pos, TTensor3D &h_neg,
                                  TTensor3D &h_persistent, int cd_step ){
            TTensor1D & h_bias = layers.back().h_bias;
            TTensor1D & v_bias = layers.back().v_bias;
            TTensor4D & W      = layers.back().W;            
            const ICRBMNode *h_node  = layers.back().h_node;
            const ICRBMNode *v_node  = layers.back().v_node;
            // go up
            tensor::crbm::conv2_r_valid( h_pos, v_pos, W, h_bias );
            h_node->cal_mean( h_pos, h_pos );

            // negative steps
            for( int i = 0 ; i < cd_step ; i ++ ){
                TTensor3D &hh = ( i == 0 ? h_persistent : h_neg );
                // sample h
                h_node->sample( h_neg, hh );

                // go down
                tensor::crbm::conv2_full( v_neg, h_neg, W, v_bias );
                v_node->cal_mean( v_neg, v_neg );

                if( param.sample_v_neg != 0 ){
                    v_node->sample( v_neg, v_neg );
                }

                // refill edge area with v_pos inorder to avoid edge effect
                if( param.refill_edge_area ){
                    tensor::crbm::refill_edge_area( v_neg, v_pos, W.y_max-1, W.x_max-1 );
                }
                
                // go up
                tensor::crbm::conv2_r_valid( h_neg, v_neg, W, h_bias );
                h_node->cal_mean( h_neg, h_neg );
            }                                    
        }

        // calculate sparse gradient and store in h_sum_mf
        inline void cal_sparse(){
            h_sum_mf  *= (1.0f/(param.batch_size*h_size));
            h_sum_mf  += -param.sparse_level;                
            switch( param.sparse_reg_method  ){
            case 0:
                {
                    h_sum_mf   = h_sum_mf * h_sum_mf_grad;
                    // leave out h_size
                    h_sum_mf  *= 2*param.sparse_lambda;                               
                    break;
                } 
            case 1:
                {
                    h_sum_mf  *= 2*param.sparse_lambda * param.batch_size * h_size;
                }
            }

            // reset regularization of non sparse node to 0
            if( param.num_non_sparse_node > 0 ){                
                h_sum_mf.x_max = param.num_non_sparse_node;
                h_sum_mf = 0.0f;
                h_sum_mf.x_max = h_sum_mf_grad.x_max;
            }
        }

        // update the weight of the last layer
        inline void update_weight(){
            TTensor1D & h_bias = layers.back().h_bias;
            TTensor1D & v_bias = layers.back().v_bias;
            TTensor4D & W      = layers.back().W;

            const float eta   = param.learning_rate/(param.batch_size*h_size);

            if( param.chg_hidden_bias ){
                // calculate sparse grad
                cal_sparse();

                // add sparse regularization and weight decay, we remember both in momentum
                if( param.use_sparse_momentum ){
                    h_bias += ( d_h_bias -= h_bias * param.wd_h + h_sum_mf * 1.0f ) * (eta * param.h_learning_rate);
				}else{
                    h_bias += ( d_h_bias -= h_bias * param.wd_h ) * (eta * param.h_learning_rate);
                    h_bias += h_sum_mf * ( - eta * param.h_learning_rate);
                }

                d_h_bias *= param.momentum;                
                h_sum_mf = 0.0f; h_sum_mf_grad = 0.0f;
            }

            if( param.chg_visible_bias ){
				if( param.v_average ){
                    // use average method to update visible bias
                    float eta_v = param.learning_rate /(param.batch_size*vv_size);
                    v_bias += ( d_v_bias-= v_bias*param.wd_v ) * ( eta_v * param.v_learning_rate );
                }else{
                    v_bias += ( d_v_bias-= v_bias*param.wd_v ) * ( eta * param.v_learning_rate );
                }
                d_v_bias *= param.momentum;
            }
 
            // use group regularization
            if( param.use_group_reg ){
                tensor::crbm::sum_2D( wd_sum , W );            
                // reset light node to 0 regularization
                if( param.num_light_node > 0 ){
                    wd_sum.x_max = param.num_light_node;
                    wd_sum = 0.0f;
                    wd_sum.x_max = d_W.z_max; 
                }
            }            
            W   += ( d_W -= W * param.wd_W ) * eta;            
            
            if( param.use_group_reg ){
                tensor::crbm::sadd__scale( W, wd_sum, -param.wd_Wsum*eta );
            }
            
            d_W *= param.momentum;
        }

        // update in training
        inline void train_update(){
            TTensor3D &v_pos = layers.back().v_state;
            TTensor3D &h_pos = layers.back().h_state;

            // whether can be use persistent chain
            TTensor3D &hp = persistent_ok ? h_neg : h_pos;
			cal_cd_steps( v_pos, v_neg, h_pos, h_neg, hp, this->cd_step );
            persistent_ok = ( param.persistent_cd !=0 );

            // this is not necessary, we add it anyway 
            if( state_counter ){
                if( param.chg_hidden_bias ){
                    d_h_bias += sum_2D( h_pos );
                    d_h_bias -= sum_2D( h_neg );         
                    layers.back().sparse_reg( h_sum_mf, h_sum_mf_grad );
                }
                if( param.chg_visible_bias ){
                    d_v_bias += sum_2D( v_pos );
                    d_v_bias -= sum_2D( v_neg );
                }                
                // calculate the gradient            
                tensor::crbm::sadd__conv2_r_big_filter( d_W, v_pos, h_pos );
                tensor::crbm::ssub__conv2_r_big_filter( d_W, v_neg, h_neg );
            }else{
                if( param.chg_hidden_bias ){
                    d_h_bias -= sum_2D( h_neg );         
                    d_h_bias += sum_2D( h_pos );
                    layers.back().sparse_reg( h_sum_mf, h_sum_mf_grad );
                }
                if( param.chg_visible_bias ){
                    d_v_bias -= sum_2D( v_neg );
                    d_v_bias += sum_2D( v_pos );
                }                
                // calculate the gradient            
                tensor::crbm::ssub__conv2_r_big_filter( d_W, v_neg, h_neg );
                tensor::crbm::sadd__conv2_r_big_filter( d_W, v_pos, h_pos );
            }
            
            state_counter = !state_counter;
            if( ++sample_counter == param.batch_size ){
                update_weight();
                sample_counter = 0;
            }
        }
        
        inline void setup_input( const apex_tensor::CTensor3D &data ){
            tensor::crbm::copy_fit( layers[0].v_state , data );
            for( size_t i = 1 ; i < layers.size() ; i ++ ){
                layers[i-1].feed_forward( layers[i].v_state );
            }  
        }
    
    public:
        virtual void get_feed_bound( int &f_z_max, int &f_y_max, int &f_x_max ){
            f_y_max = layers.back().v_state.y_max; 
            f_x_max = layers.back().v_state.x_max; 
            f_z_max = layers.back().h_state.z_max; 
            layers.back().reget_bound( f_y_max, f_x_max );            
        }

        virtual void feed_forward_trunk( apex_tensor::CTensor4D &state_out, const apex_tensor::CTensor4D &data ){            
            int y_max = layers.back().v_state.y_max; 
            int x_max = layers.back().v_state.x_max; 
            int h_max = layers.back().h_state.z_max; 
            layers.back().reget_bound( y_max, x_max );
             
            TTensor3D v_state( h_max, y_max, x_max );
            tensor::alloc_space( v_state );
            
            for( int i = 0 ; i < data.h_max ; i ++ ){
                setup_input( data[i] );
                layers.back().feed_forward( v_state );

                apex_tensor::CTensor3D so = state_out[i];
                tensor::copy( so, v_state );
            }            
            tensor::free_space( v_state );
        }

        virtual void train_update( const apex_tensor::CTensor3D &data ){
            setup_input( data );
            train_update();
        }

        virtual void train_update_trunk( const apex_tensor::CTensor4D &data ){
            for( int i = 0 ; i < data.h_max ; i ++ )
                train_update( data[i] );
        }
        
        // do validation, return the statistics
        virtual void validate_stats( CRBMModelStats &stats, const apex_tensor::CTensor4D &data ){
            TTensor4D grad_W;
            TTensor1D pos_grad_h, neg_grad_h, pos_grad_v, neg_grad_v, loss, grad_sparse;
            grad_W     = clone( stats.grad_W );
            pos_grad_h = clone( stats.pos_grad_h );
            neg_grad_h = clone( stats.neg_grad_h );
            pos_grad_v = clone( stats.pos_grad_v );
            neg_grad_v = clone( stats.neg_grad_v );
            loss       = clone( stats.loss );
            grad_sparse= clone( stats.grad_sparse );
            
            TTensor3D &v_pos = layers.back().v_state;                
            TTensor3D &h_pos = layers.back().h_state;     
                        
            for( int i = 0 ; i < data.h_max ; i ++ ){
                setup_input( data[i] );                

                cal_cd_steps( v_pos, v_neg, h_pos, h_neg, h_pos, 1 );
                
                layers.back().sparse_reg( h_sum_mf, h_sum_mf_grad );

                tensor::crbm::sadd__conv2_r_big_filter( grad_W, v_pos, h_pos );
                tensor::crbm::ssub__conv2_r_big_filter( grad_W, v_neg, h_neg );                

                pos_grad_h += sum_2D( h_pos );
                neg_grad_h -= sum_2D( h_neg );               
                pos_grad_v += sum_2D( v_pos );
                neg_grad_v -= sum_2D( v_neg );
                v_neg      -= v_pos;
                v_neg       = v_neg * v_neg;

                loss += sum_2D( v_neg );
				if( i % param.batch_size == param.batch_size - 1 ){ 
					cal_sparse();	
					grad_sparse -= h_sum_mf;
					h_sum_mf = 0.0f; h_sum_mf_grad = 0.0f;
				}
            }                 
            stats.h_size  = h_size;
            stats.v_size  = v_size;
            stats.vv_size = vv_size;
            stats.sample_counter += data.h_max;
                        
            tensor::copy( stats.grad_W, grad_W );
            tensor::copy( stats.pos_grad_h, pos_grad_h );
            tensor::copy( stats.neg_grad_h, neg_grad_h );
            tensor::copy( stats.pos_grad_v, pos_grad_v );
            tensor::copy( stats.neg_grad_v, neg_grad_v );
            tensor::copy( stats.loss, loss );
            tensor::copy( stats.grad_sparse, grad_sparse );
            
            tensor::free_space( grad_W );
            tensor::free_space( pos_grad_h );
            tensor::free_space( neg_grad_h );
            tensor::free_space( pos_grad_v );
            tensor::free_space( neg_grad_v );
            tensor::free_space( loss );
            tensor::free_space( grad_sparse );
        }
       
        /* clone model trainied to model */
        virtual void clone_model( CDBNModel &model )const{
            if( model.layers.size() != layers.size() ){
                printf("error model size\n"); exit( -1 );
            } 

            CRBMModel &md = model.layers.back();
            const CRBMLayer &mm = layers.back();
            
            tensor::copy( md.W , mm.W );
            tensor::copy( md.h_bias , mm.h_bias );
            tensor::copy( md.v_bias , mm.v_bias );
            tensor::copy( md.d_W , d_W );
            tensor::copy( md.d_h_bias , d_h_bias );
            tensor::copy( md.d_v_bias , d_v_bias );            
        }       

        /* set steps of CD */
        virtual void set_cd_step( int cd_step ){
            this->cd_step = cd_step;
        }                
    };

    namespace factory{
        // create a stacked rbm
        ICRBM *create_crbm( const CDBNModel &model, const CRBMTrainParam &param ){
            return  new CRBMSimple( model, param );
        }
    };
};

#endif

