#ifndef _APEX_CRBM_CPP_
#define _APEX_CRBM_CPP_

#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "apex_crbm.h"
#include "../apex_crbm_model.h"
#include "../apex_crbm_model_stats.h"
#include "../../tensor/apex_tensor.h"
#include "../../utils/apex_utils.h"
#include <vector>

namespace apex_rbm{
    using namespace std;
    using namespace apex_tensor;    
    using namespace apex_utils::iterator;

    // node of CRBM
    class ICRBMNode{
    public:
        // get mean value given energy
        virtual void cal_mean( TTensor3D &mean , const TTensor3D &energy )const = 0;
        // sample directly from energy 
        virtual void sample_from_energy( TTensor3D &state, const TTensor3D &energy )const = 0;
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
    public:
        virtual ~ICRBMNode(){}
   };

    // bianry node
    class CRBMBinaryNode : public ICRBMNode{
    public:
        virtual void sample_from_energy( TTensor3D &state, const TTensor3D &energy )const{
            cal_mean( state, energy ); 
            state = sample_binary( state );
        }
        virtual void cal_mean( TTensor3D &mean , const TTensor3D &energy )const{
            mean = sigmoid( energy );
        }               
        virtual void feed_forward( TTensor3D &v_next, const TTensor3D &h_curr )const{
            tensor::crbm::copy_fit( v_next, h_curr );
        }
        
        virtual void forward_bias( TTensor1D &v_bias_next, const TTensor1D &h_bias_curr )const{
            tensor::copy( v_bias_next, h_bias_curr );
        }

        virtual void reget_bound ( int &input_y_max, int &input_x_max  )const{}       
        virtual void reget_hidden_bound( int &h_y_max, int &h_x_max )const{}
        virtual void sparse_reg( TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad, const TTensor3D &h_pos )const{
            tensor::crbm::add_sparse_info( h_sum_mf, h_sum_mf_grad, h_pos , 1 );
        }
    };

    // ReLU node
    template<bool scale_energy>
    class CRBMReLUNode : public ICRBMNode{
    private:
        TENSOR_FLOAT energy_scale;
    public :
        CRBMReLUNode( TENSOR_FLOAT energy_scale = 1.0f ){
            this->energy_scale = energy_scale;
        }
        virtual void sample_from_energy( TTensor3D &state, const TTensor3D &energy )const{
            if( scale_energy ) state = energy * energy_scale;
            tensor::rbm::sample_recified_linear( state, state );
        }
        virtual void cal_mean( TTensor3D &mean , const TTensor3D &energy )const{
            if( scale_energy ) mean = energy * energy_scale;
            tensor::rbm::mean_recified_linear( mean, mean );
        }               
        virtual void feed_forward( TTensor3D &v_next, const TTensor3D &h_curr )const{
            tensor::crbm::copy_fit( v_next, h_curr );
        }
        
        virtual void forward_bias( TTensor1D &v_bias_next, const TTensor1D &h_bias_curr )const{
            tensor::copy( v_bias_next, h_bias_curr );
        }
        virtual void reget_bound ( int &input_y_max, int &input_x_max  )const{}       
        virtual void reget_hidden_bound( int &h_y_max, int &h_x_max )const{}
        virtual void sparse_reg( TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad, const TTensor3D &h_pos )const{
            // for temporary use
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
        virtual void sample_from_energy( TTensor3D &state, const TTensor3D &energy )const{
            cal_mean( state, energy );
            state = sample_gaussian( state, sigma );
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

        virtual void reget_bound ( int &input_y_max, int &input_x_max  )const{}        
        virtual void reget_hidden_bound( int &h_y_max, int &h_x_max )const{}

        virtual void sparse_reg( TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad, const TTensor3D &h_pos )const{
            apex_utils::error("gaussian hidden is not supported");
        }
    };
    

    // maxpooling node 
    template<bool scale_energy>
    class CRBMMaxpoolNode : public ICRBMNode{
    private:
        int pool_size;
        TENSOR_FLOAT energy_scale;
    public:
        CRBMMaxpoolNode( int pool_size, TENSOR_FLOAT energy_scale = 1.0f ){
            this->pool_size = pool_size;
            this->energy_scale = energy_scale;
        }
        virtual void sample_from_energy( TTensor3D &state, const TTensor3D &energy )const{
            cal_mean( state, energy );
            tensor::crbm::sample_maxpooling_2D( state, state, pool_size );
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
			v_bias_next += (TENSOR_FLOAT)( 2.0 * log((double)pool_size) );
        }
        virtual void feed_forward( TTensor3D &v_next, const TTensor3D &h_curr )const{
            tensor::crbm::pool_up( v_next, h_curr, pool_size );
        }
        // reget the bound of data
        virtual void reget_bound ( int &input_y_max, int &input_x_max )const{
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
        case model_type::BINARY_MAXPOOL    : return new CRBMBinaryNode();
        case model_type::GAUSSIAN_MAXPOOL_A: return new CRBMGaussianNode<false>( param.v_sigma );
        case model_type::GAUSSIAN_MAXPOOL_B: return new CRBMGaussianNode<true> ( param.v_sigma );
        case model_type::GAUSSIAN_RELU_B   : return new CRBMGaussianNode<true> ( param.v_sigma );
        case model_type::BINARY_RELU       : return new CRBMBinaryNode();
        default: return NULL;
        }
    }
    inline ICRBMNode *create_hidden_node( const CRBMModelParam &param ){
        switch( param.model_type ){
        case model_type::BINARY_MAXPOOL:     return new CRBMMaxpoolNode<false>( param.pool_size );
        case model_type::GAUSSIAN_MAXPOOL_A: return new CRBMMaxpoolNode<false>( param.pool_size );
        case model_type::GAUSSIAN_MAXPOOL_B: return new CRBMMaxpoolNode<true> ( param.pool_size, 1.0f/(param.v_sigma*param.v_sigma) );
        case model_type::GAUSSIAN_RELU_B   : return new CRBMReLUNode<true>    ( 1.0f/(param.v_sigma*param.v_sigma) );
        case model_type::BINARY_RELU       : return new CRBMReLUNode<false>   ();
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
            this->v_node = create_visible_node( model.param );
            this->h_node = create_hidden_node ( model.param );

            this->W      = clone( model.W );
            this->h_bias = clone( model.h_bias );
            this->v_bias = clone( model.v_bias );          

            // we fit the input size to ensure perfect pooling
            v_state.z_max = model.param.v_max;
            h_state.z_max = model.param.h_max;
            this->refit_node_size( y_max, x_max );
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

        // refit the node size to input size, the size must not exceed max size 
        // whether the input size is invalid
        inline bool refit_node_size( int input_y_max, int input_x_max ){
            if( input_y_max != v_state.y_max || input_x_max != v_state.x_max ){
                int h_y_max = input_y_max - W.y_max + 1;
                int h_x_max = input_x_max - W.x_max + 1;
                h_node->reget_hidden_bound( h_y_max, h_x_max );
                if( h_y_max <= 0 || h_x_max <=0 ) return false;
                v_state.set_param( v_state.z_max, h_y_max+W.y_max-1, h_x_max+W.x_max-1 );
                h_state.set_param( h_state.z_max, h_y_max, h_x_max );
            }
            return true;
        }

        inline void forward_bias( TTensor1D &v_bias_next ) const{
            h_node->forward_bias( v_bias_next, h_bias );
        }

        // feed forward to next state
        inline void feed_forward( TTensor3D &v_state_next ){
            tensor::crbm::conv2_r_valid( h_state, v_state, W, h_bias );
            h_node->cal_mean( h_state, h_state );
            h_node->feed_forward( v_state_next, h_state );
        }        

        // reget bound of output
        inline void reget_bound( int &y_max, int &x_max ) const{
            y_max = y_max - W.y_max + 1;  
            x_max = x_max - W.x_max + 1;  
            h_node->reget_bound( y_max, x_max );
        }

        // sparse regularization
        inline void sparse_reg( TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad ) const{
            h_node->sparse_reg( h_sum_mf, h_sum_mf_grad, h_state );
        }
    };   
    
    // trainer of CRBM
    class CRBMLightTrainer: public ICRBMTrainer{       
    private:
        int cd_step;
        CRBMTrainParam param;
        bool validation_init;
        int  h_size, v_size, vv_size;
    private:
        int sample_counter, state_counter;
    private:
        TTensor4D d_W;
        TTensor2D reg_group_W;
        TTensor1D d_h_bias, d_v_bias;
        TTensor1D h_sum_mf, h_sum_mf_grad, d_h_sparse, v_sum_mf;
    private:
        CRBMLayer layer;
        TTensor3D v_neg,  h_neg;
    private:
        inline void sync_size_to_layer( bool force_sync = false ){
            if( force_sync || layer.h_state.y_max  != h_neg.y_max || layer.h_state.x_max != h_neg.x_max ){
                h_neg.set_param( h_neg.z_max, layer.h_state.y_max, layer.h_state.x_max );
                v_neg.set_param( v_neg.z_max, layer.v_state.y_max, layer.v_state.x_max );

                h_size   = layer.h_state.y_max * layer.h_state.x_max;
                v_size   = layer.v_state.y_max * layer.v_state.x_max;
                if( param.refill_edge_area ){
                    vv_size = (layer.h_state.y_max-d_W.y_max+1) * (layer.h_state.x_max-d_W.x_max+1);	
                }else{
                    vv_size = v_size;
                }
            }
        }
        inline void init_data( const CRBMModel & model ){
            validation_init = false; 
            
            d_W      = clone( model.d_W );
            d_h_bias = clone( model.d_h_bias );
            d_v_bias = clone( model.d_v_bias );

            d_h_sparse    = alloc_like( d_h_bias );
            h_sum_mf      = alloc_like( d_h_bias );
            v_sum_mf      = alloc_like( d_v_bias );
            h_sum_mf_grad = alloc_like( d_h_bias );
            v_neg         = alloc_like( layer.v_state );
            h_neg         = alloc_like( layer.h_state );
            
            sample_counter = 0; state_counter = 0; 
            h_sum_mf = 0.0f; h_sum_mf_grad = 0.0f; v_sum_mf = 0.0f;

            reg_group_W.set_param( model.param.v_max, model.param.h_max );
            tensor::alloc_space( reg_group_W );
            
            this->sync_size_to_layer( true );
        }
        // initialize the asynchronize 
        // stream engine to support aynchronize speed up
        inline void init_async(){
            init_stream_engine( 3 );                   
            TTensor1D & h_bias = layer.h_bias;
            TTensor1D & v_bias = layer.v_bias;
            TTensor4D & W      = layer.W;            
            async::set_dependecy( d_v_bias, 1 );
            async::set_dependecy(   v_bias, 1 );
            async::set_dependecy( d_h_bias, 2 );
            async::set_dependecy(   h_bias, 2 );
            async::set_dependecy( d_h_sparse, 2 );
            async::set_dependecy( h_sum_mf, 2 );
            async::set_dependecy( h_sum_mf_grad, 2 );
            async::set_dependecy( d_W     , 3 );
            async::set_dependecy(   W     , 3 );
        }        
    public:
        CRBMLightTrainer( const CRBMModel &model, const CRBMTrainParam &param )
            :layer( model, param.input_y_max, param.input_x_max ){
            init_tensor_engine( 0 );
            this->param = param;
            init_data( model );            
            init_async();
            printf("CRBMLightTrainer initialized, h_size=%d, v_size=%d, vv_size=%d\n", h_size, v_size, vv_size );
        }
        virtual ~CRBMLightTrainer(){
            layer.free_space();
            tensor::free_space( d_h_bias );
            tensor::free_space( d_v_bias );
            tensor::free_space( d_W );
            tensor::free_space( v_sum_mf );
            tensor::free_space( d_h_sparse );
            tensor::free_space( h_sum_mf );
            tensor::free_space( h_sum_mf_grad );
            tensor::free_space( v_neg );
            tensor::free_space( h_neg );
            tensor::free_space( reg_group_W );
            destroy_validation_data();

            // destroy the tensor engine
            destroy_tensor_engine();
            destroy_stream_engine();          
        }              
    private:
        // calculate the datas in cd steps
        inline void cal_cd_steps( TTensor3D &v_pos, TTensor3D &v_neg, 
                                  TTensor3D &h_pos, TTensor3D &h_neg,
                                  TTensor3D &h_persistent, int cd_step ){
            TTensor1D & h_bias = layer.h_bias;
            TTensor1D & v_bias = layer.v_bias;
            TTensor4D & W      = layer.W;            
            const ICRBMNode *h_node  = layer.h_node;
            const ICRBMNode *v_node  = layer.v_node;
            // go up
            tensor::crbm::conv2_r_valid( h_pos, v_pos, W, h_bias );

            // negative steps
            for( int i = 0 ; i < cd_step ; i ++ ){
                TTensor3D &hh = ( i == 0 ? h_persistent : h_neg );
                // sample h
                h_node->sample_from_energy( h_neg, hh );

                // go down
                tensor::crbm::conv2_full( v_neg, h_neg, W, v_bias );

                if( param.sample_v_neg ){
                    v_node->sample_from_energy( v_neg, v_neg );
                }else{
                    v_node->cal_mean( v_neg, v_neg );
                }

                // refill edge area with v_pos inorder to avoid edge effect
                if( param.refill_edge_area ){
                    tensor::crbm::refill_edge_area( v_neg, v_pos, W.y_max-1, W.x_max-1 );
                }
                
                // go up
                tensor::crbm::conv2_r_valid( h_neg, v_neg, W, h_bias );
            }                                    
            h_node->cal_mean( h_neg, h_neg );
            h_node->cal_mean( h_pos, h_pos );            
        }

        // calculate sparse gradient and store in h_sum_mf
        inline void cal_sparse(){
            {// combine sparse memory using sparse ratio
                h_sum_mf *= (1.0f-param.sparse_mem_ratio); 
                h_sum_mf_grad *= (1.0f-param.sparse_mem_ratio);                                
            }

            d_h_sparse = h_sum_mf * (1.0f/(param.batch_size*h_size));
            d_h_sparse += -param.sparse_level;                

            switch( param.sparse_reg_method  ){
            case sparse_loss::L2_LOSS :
            default:
                {
                    d_h_sparse = d_h_sparse * h_sum_mf_grad;
                    d_h_sparse *= 2*param.sparse_lambda;                               
                    break;
                }
            case sparse_loss::KL_LOSS :
                {// remember h_size will be devided by h gradient
                    d_h_sparse *= param.sparse_lambda * param.batch_size * h_size;                               
                    break;
                }                   
            }            

            {// change of sparse histogram, remember previous sparse ratio
                h_sum_mf *= param.sparse_mem_ratio / (1.0f-param.sparse_mem_ratio); 
                h_sum_mf_grad *= param.sparse_mem_ratio / (1.0f-param.sparse_mem_ratio);
            }
        }
      
        // update the weight 
        inline void update_weight(){
            TTensor1D & h_bias = layer.h_bias;
            TTensor1D & v_bias = layer.v_bias;
            TTensor4D & W      = layer.W;

            const float eta   = param.learning_rate/(param.batch_size*h_size);

            if( param.chg_hidden_bias ){                                
                // calculate sparse grad
                cal_sparse();

                h_bias += ( d_h_bias -= h_bias * param.wd_h + d_h_sparse * 1.0f ) * (eta * param.h_learning_rate);                
                d_h_bias *= param.momentum;                
                
                if( param.sparse_reg_edge ){
                    v_sum_mf *= 1.0f / (param.batch_size*v_size);
                    // add sparse panelty to edge 
                    reg_group_W = dot( v_sum_mf.T(), d_h_sparse );
                    tensor::crbm::sadd__scale( d_W, reg_group_W, -1.0f );
                    // reset v_sum_mf
                    v_sum_mf = 0.0f;
                }
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
                        
            W   += ( d_W -= W * param.wd_W ) * eta;                        
            d_W *= param.momentum;           
        }

        // update in training
        inline void train_update(){
            TTensor3D &v_pos = layer.v_state;
            TTensor3D &h_pos = layer.h_state;

            // whether can be use persistent chain
			cal_cd_steps( v_pos, v_neg, h_pos, h_neg, h_pos, this->cd_step );

            // if we need to regularize edge sparsity, we need this information
            if( param.sparse_reg_edge ) v_sum_mf += sum_2D( v_pos );

            // this is not necessary, we add it anyway 
            if( state_counter ){
                if( param.chg_hidden_bias ){
                    d_h_bias += sum_2D( h_pos );
                    d_h_bias -= sum_2D( h_neg );         
                    layer.sparse_reg( h_sum_mf, h_sum_mf_grad );                                        
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
                    layer.sparse_reg( h_sum_mf, h_sum_mf_grad );
                }
                if( param.chg_visible_bias ){
                    d_v_bias -= sum_2D( v_neg );
                    d_v_bias += sum_2D( v_pos );
                }                
                // calculate the gradient            
                tensor::crbm::ssub__conv2_r_big_filter( d_W, v_neg, h_neg );
                tensor::crbm::sadd__conv2_r_big_filter( d_W, v_pos, h_pos );
            }
            // revert state counter
            state_counter = !state_counter;
            if( ++sample_counter == param.batch_size ){
                update_weight();
                sample_counter = 0;
            }
        }        
    private:
        inline void setup_input( const CTensor3D &data ){
            if( layer.refit_node_size( data.y_max, data.x_max ) ) {
                this->sync_size_to_layer();
                tensor::crbm::copy_fit( layer.v_state, data );            
            }else{
                printf("\ninvalid input size:y_max=%d,x_max=%d\n", data.y_max, data.x_max );
                apex_utils::error("invalid input size");
            }
        }
        
    private:
        // data structs for validation
        TTensor4D grad_W;
        TTensor1D pos_grad_h, neg_grad_h, pos_grad_v, neg_grad_v, loss, grad_sparse;
        inline void init_validation_data(){
            if( !validation_init ){
                grad_W     = alloc_like( d_W );
                pos_grad_h = alloc_like( d_h_bias );
                neg_grad_h = alloc_like( d_h_bias );
                pos_grad_v = alloc_like( d_v_bias );
                neg_grad_v = alloc_like( d_v_bias );
                loss       = alloc_like( d_v_bias );
                grad_sparse= alloc_like( d_h_bias );
                validation_init = true;
            }
        }
        inline void destroy_validation_data(){
            if( validation_init ){
                tensor::free_space( grad_W );
                tensor::free_space( pos_grad_h );
                tensor::free_space( neg_grad_h );
                tensor::free_space( pos_grad_v );
                tensor::free_space( neg_grad_v );
                tensor::free_space( loss );
                tensor::free_space( grad_sparse );
            }
        }
    public:                      
        virtual void clone_model( CRBMModel & md )const{           
            tensor::copy( md.W , layer.W );
            tensor::copy( md.h_bias , layer.h_bias );
            tensor::copy( md.v_bias , layer.v_bias );
            tensor::copy( md.d_W , d_W );
            tensor::copy( md.d_h_bias , d_h_bias );
            tensor::copy( md.d_v_bias , d_v_bias );            
        }   

        virtual void train_update( const CTensor3D &data ){ 
            this->setup_input( data );
            this->train_update();
        }

        virtual void set_cd_step( int cd_step ){
            this->cd_step = cd_step;
        }

        virtual void validate_stats( CRBMModelStats &stats, IIterator<CTensor3D> *data_itr ){
            init_validation_data();

            tensor::copy( grad_W    , stats.grad_W );
            tensor::copy( pos_grad_h, stats.pos_grad_h );
            tensor::copy( neg_grad_h, stats.neg_grad_h );
            tensor::copy( pos_grad_v, stats.pos_grad_v );
            tensor::copy( neg_grad_v, stats.neg_grad_v );
            tensor::copy( loss      , stats.loss );
            tensor::copy( grad_sparse, stats.grad_sparse );           

            TTensor3D &v_pos = layer.v_state;                
            TTensor3D &h_pos = layer.h_state;     
            
            int counter = 0;

            data_itr->before_first();
            while( data_itr->next() ){
                setup_input( data_itr->value() );                

                cal_cd_steps( v_pos, v_neg, h_pos, h_neg, h_pos, 1 );
                
                layer.sparse_reg( h_sum_mf, h_sum_mf_grad );

                tensor::crbm::sadd__conv2_r_big_filter( grad_W, v_pos, h_pos );
                tensor::crbm::ssub__conv2_r_big_filter( grad_W, v_neg, h_neg );                

                pos_grad_h += sum_2D( h_pos );
                neg_grad_h -= sum_2D( h_neg );               
                pos_grad_v += sum_2D( v_pos );
                neg_grad_v -= sum_2D( v_neg );
                v_neg      -= v_pos;
                v_neg       = v_neg * v_neg;

                loss += sum_2D( v_neg );
				if( counter ++ % param.batch_size == param.batch_size - 1 ){ 
                    cal_sparse();
                    grad_sparse -= d_h_sparse;
				}
            }                 

            h_sum_mf = 0.0f; h_sum_mf_grad = 0.0f;
            stats.h_size  = h_size;
            stats.v_size  = v_size;
            stats.vv_size = vv_size;
            stats.sample_counter += counter;
                        
            tensor::copy( stats.grad_W, grad_W );
            tensor::copy( stats.pos_grad_h, pos_grad_h );
            tensor::copy( stats.neg_grad_h, neg_grad_h );
            tensor::copy( stats.pos_grad_v, pos_grad_v );
            tensor::copy( stats.neg_grad_v, neg_grad_v );
            tensor::copy( stats.loss, loss );
            tensor::copy( stats.grad_sparse, grad_sparse );           
        }
    }; 
    
    class CRBMLightInferencer: public ICRBMInferencer{
    private:
        CTensor1D top_bias_fwd;
        TTensor3D top_state;
        vector<CRBMLayer> layers;        
    public:
        CRBMLightInferencer( const CDBNModel &model, int input_y_max, int input_x_max ){
            init_tensor_engine( 0 );
            
            for( size_t i = 0 ; i < model.layers.size() ; i ++ ){                
                layers.push_back( CRBMLayer( model.layers[i], input_y_max, input_x_max ) );
                layers.back().reget_bound( input_y_max, input_x_max );
            }
            // top state 
            top_state.set_param( model.layers.back().param.h_max, input_y_max, input_x_max );
            tensor::alloc_space( top_state );        
            
            set_bias_fwd();
        }
        ~CRBMLightInferencer(){
            for( size_t i = 0 ; i < layers.size() ; i ++ )
                layers[i].free_space();
            layers.clear();
            tensor::free_space( top_bias_fwd );
            tensor::free_space( top_state );
            destroy_tensor_engine();
        }

        inline void set_bias_fwd(){
            TTensor1D vb;
            vb = alloc_like( layers.back().h_bias );
            layers.back().forward_bias( vb );
            
            top_bias_fwd.set_param( vb.x_max );
            tensor::alloc_space( top_bias_fwd );
            tensor::copy( top_bias_fwd, vb );
            tensor::free_space( vb );
        }
    public:
        virtual void set_input( const CTensor3D &data ){                                
            int y_max = data.y_max;
            int x_max = data.x_max; 

            if( y_max != layers[0].v_state.y_max || 
                x_max != layers[0].v_state.x_max ){
                for( size_t i = 0; i < layers.size() ; i ++ ){
                    if( layers[i].refit_node_size( y_max, x_max ) ){ 
                        layers[i].reget_bound( y_max, x_max );
                    }else{
                        apex_utils::error("invalid node size");
                    }
                } 
                top_state.set_param( top_state.z_max, y_max, x_max );
            }            
            tensor::crbm::copy_fit( layers[0].v_state , data );
        }
        virtual void get_top_bound( int &t_z_max, int &t_y_max, int &t_x_max ) const{
            t_z_max = top_state.z_max;
            t_y_max = top_state.y_max;
            t_x_max = top_state.x_max;
        } 
        virtual void infer_top_layer( CTensor3D &dout ){
            for( size_t i = 1 ; i < layers.size() ; i ++ ){
                layers[i-1].feed_forward( layers[i].v_state );
            }                       
            layers.back().feed_forward( top_state );
            tensor::copy( dout, top_state );
        }
        virtual void forward_bias( apex_tensor::CTensor1D &v_bias_next ) const{
            tensor::copy( v_bias_next, top_bias_fwd );
        }
    };

    namespace factory{
        ICRBMTrainer    *create_crbm_trainer   ( const CRBMModel &model, const CRBMTrainParam &param ){
            return new CRBMLightTrainer( model, param );
        }
        ICRBMInferencer *create_crbm_inferencer( const CDBNModel &model, int input_y_max, int input_x_max ){
            return new CRBMLightInferencer( model, input_y_max, input_x_max );
        }
    };
};

#endif

