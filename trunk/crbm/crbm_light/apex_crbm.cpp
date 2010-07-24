#ifndef _APEX_CRBM_CPP_
#define _APEX_CRBM_CPP_

#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "apex_crbm.h"
#include "../apex_crbm_model.h"
#include "../apex_crbm_model_stats.h"
#include "../../tensor/apex_tensor.h"

#include <vector>

namespace apex_rbm{
    using namespace std;
    using namespace apex_tensor;    

    // interface node of CRBM
    class ICRBMNode{
    protected:
        // virtual function table
        // sample state from mean value 
        void (*fp_sample)      ( const ICRBMNode *p_self, TTensor3D &state, const TTensor3D &mean   );
        // calculate mean value from energy
        void (*fp_cal_mean)    ( const ICRBMNode *p_self, TTensor3D &mean , const TTensor3D &energy );
        // feed forward data needed   
        void (*fp_feed_forward)( const ICRBMNode *p_self, TTensor3D &v_next, const TTensor3D &h_curr );
        // feed forward bias to next layer 
        void (*fp_forward_bias)( const ICRBMNode *p_self, TTensor1D &v_bias_next, const TTensor1D &h_bias_curr );
        // reget the bound of data
        void (*fp_reget_bound) ( const ICRBMNode *p_self, int &input_y_max, int &input_x_max );
        // reget the bound of hidden data 
        void (*fp_reget_hidden_bound)( const ICRBMNode *p_self, int &h_y_max, int &h_x_max );
        // calculate sparse_regularization
        void (*fp_sparse_reg)        ( const ICRBMNode *p_self, TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad, const TTensor3D &h_pos );
    protected:
        ICRBMNode(){}
    public:
        // virtual destructor for safe destruction
        virtual ~ICRBMNode(){}
    public:
        // rejump
        inline void sample  ( TTensor3D &state, const TTensor3D &mean ) const{
            (*fp_sample)( this, state, mean );
        }
        inline void cal_mean( TTensor3D &mean , const TTensor3D &energy ) const{
            (*fp_cal_mean)( this, mean, energy );
        }
        inline void feed_forward( TTensor3D &v_next, const TTensor3D &h_curr ) const{
            (*fp_feed_forward)( this, v_next, h_curr );
        }
        inline void forward_bias( TTensor1D &v_bias_next, const TTensor1D &h_bias_curr ) const{
            (*fp_forward_bias)( this, v_bias_next, h_bias_curr );
        }
        inline void reget_bound ( int &input_y_max, int &input_x_max  ) const{
            (*fp_reget_bound)( this, input_y_max, input_x_max );
        }
        inline void reget_hidden_bound( int &h_y_max, int &h_x_max ) const{
            (*fp_reget_hidden_bound)( this, h_y_max, h_x_max );
        }
        inline void sparse_reg( TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad, const TTensor3D &h_pos ) const{
            (*fp_sparse_reg)( this, h_sum_mf, h_sum_mf_grad, h_pos );
        }
    };

    // bianry node
    class CRBMBinaryNode : public ICRBMNode{
    private:
        static void s_sample( const ICRBMNode *p_self, TTensor3D &state, const TTensor3D &mean ){
            state = sample_binary( mean );
        }

        static void s_cal_mean( const ICRBMNode *p_self, TTensor3D &mean , const TTensor3D &energy ){
            mean = sigmoid( energy );
        }  

        static void s_feed_forward( const ICRBMNode *p_self, TTensor3D &v_next, const TTensor3D &h_curr ){
            tensor::crbm::copy_fit( v_next, h_curr );
        }

        static void s_forward_bias( const ICRBMNode *p_self, TTensor1D &v_bias_next, const TTensor1D &h_bias_curr ){
            tensor::copy( v_bias_next, h_bias_curr );
        }

        static void s_reget_bound ( const ICRBMNode *p_self, int &input_y_max, int &input_x_max  ){}
        static void s_reget_hidden_bound( const ICRBMNode *p_self, int &h_y_max, int &h_x_max ){}

        static void s_sparse_reg( const ICRBMNode *p_self, TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad, const TTensor3D &h_pos ){
            tensor::crbm::add_sparse_info( h_sum_mf, h_sum_mf_grad, h_pos , 1 );
        }
    public:
        CRBMBinaryNode(){
            // load virtual function tables
            this->fp_sample       = s_sample;
            this->fp_cal_mean     = s_cal_mean;
            this->fp_feed_forward = s_feed_forward;
            this->fp_forward_bias = s_forward_bias; 
            this->fp_reget_bound  = s_reget_bound;
            this->fp_reget_hidden_bound = s_reget_hidden_bound;
            this->fp_sparse_reg   = s_sparse_reg;
        }
    };

    template<bool use_type2>
    class CRBMGaussianNode : public ICRBMNode{
    private:
        TENSOR_FLOAT sigma, sigma_sqr;
    private:
        static void s_sample( const ICRBMNode *p_self, TTensor3D &state, const TTensor3D &mean ){
            const CRBMGaussianNode *self = static_cast<const CRBMGaussianNode*>( p_self );
            state = sample_gaussian( mean, self->sigma );
        }

        static void s_cal_mean( const ICRBMNode *p_self, TTensor3D &mean , const TTensor3D &energy ){
            const CRBMGaussianNode *self = static_cast<const CRBMGaussianNode*>( p_self );
            if( use_type2 ){
                if( mean.elem != energy.elem ) tensor::copy( mean, energy );
            }else{
                mean =  energy * self->sigma_sqr;
            }
        }  

        static void s_feed_forward( const ICRBMNode *p_self, TTensor3D &v_next, const TTensor3D &h_curr ){
            tensor::crbm::copy_fit( v_next, h_curr );
        }

        static void s_forward_bias( const ICRBMNode *p_self, TTensor1D &v_bias_next, const TTensor1D &h_bias_curr ){
            tensor::copy( v_bias_next, h_bias_curr );
        }

        static void s_reget_bound ( const ICRBMNode *p_self, int &input_y_max, int &input_x_max  ){}
        static void s_reget_hidden_bound( const ICRBMNode *p_self, int &h_y_max, int &h_x_max ){}

        static void s_sparse_reg( const ICRBMNode *p_self, TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad, const TTensor3D &h_pos ){
            printf("currently gaussian node doesn't support sparse reg\n"); 
            exit(-1);
        }
    public:
        CRBMGaussianNode( TENSOR_FLOAT sigma ){
            this->sigma       = sigma;
            this->sigma_sqr   = sigma*sigma;
            // load virtual function tables
            this->fp_sample       = s_sample;
            this->fp_cal_mean     = s_cal_mean;
            this->fp_feed_forward = s_feed_forward;
            this->fp_forward_bias = s_forward_bias; 
            this->fp_reget_bound  = s_reget_bound;
            this->fp_reget_hidden_bound = s_reget_hidden_bound;
            this->fp_sparse_reg   = s_sparse_reg;
        }
    };

    // maxpooling node
    template<bool scale_energy>
    class CRBMMaxpoolNode : public ICRBMNode{
    private:
        int pool_size;
        TENSOR_FLOAT energy_scale;
    private:
        static void s_sample( const ICRBMNode *p_self, TTensor3D &state, const TTensor3D &mean ){
            const CRBMMaxpoolNode *self = static_cast<const CRBMMaxpoolNode*>( p_self );
            tensor::crbm::sample_maxpooling_2D( state, mean, self->pool_size );
        }

        static void s_cal_mean( const ICRBMNode *p_self, TTensor3D &mean , const TTensor3D &energy ){
            const CRBMMaxpoolNode *self = static_cast<const CRBMMaxpoolNode*>( p_self );
            if( !scale_energy ){ 
                tensor::crbm::norm_maxpooling_2D( mean, energy, self->pool_size );
            }else{
                mean = energy * self->energy_scale;
                tensor::crbm::norm_maxpooling_2D( mean, mean, self->pool_size );
            }
        }  

        static void s_feed_forward( const ICRBMNode *p_self, TTensor3D &v_next, const TTensor3D &h_curr ){
            const CRBMMaxpoolNode *self = static_cast<const CRBMMaxpoolNode*>( p_self );
            tensor::crbm::pool_up( v_next, h_curr, self->pool_size );
        }

        static void s_forward_bias( const ICRBMNode *p_self, TTensor1D &v_bias_next, const TTensor1D &h_bias_curr ){
            const CRBMMaxpoolNode *self = static_cast<const CRBMMaxpoolNode*>( p_self );
            tensor::copy( v_bias_next, h_bias_curr );
			v_bias_next += (float)( 2.0 * log((double)(self->pool_size)) );
        }

        static void s_reget_bound ( const ICRBMNode *p_self, int &input_y_max, int &input_x_max  ){
            const CRBMMaxpoolNode *self = static_cast<const CRBMMaxpoolNode*>( p_self );
            input_x_max /= self->pool_size;
            input_y_max /= self->pool_size;
        }

        static void s_reget_hidden_bound( const ICRBMNode *p_self, int &h_y_max, int &h_x_max ){
            const CRBMMaxpoolNode *self = static_cast<const CRBMMaxpoolNode*>( p_self );
            h_y_max = (h_y_max / self->pool_size) * self->pool_size;
            h_x_max = (h_x_max / self->pool_size) * self->pool_size;
        }

        static void s_sparse_reg( const ICRBMNode *p_self, TTensor1D &h_sum_mf, TTensor1D &h_sum_mf_grad, const TTensor3D &h_pos ){
            const CRBMMaxpoolNode *self = static_cast<const CRBMMaxpoolNode*>( p_self );
            tensor::crbm::add_sparse_info( h_sum_mf, h_sum_mf_grad, h_pos , self->pool_size );
        }
    public:
        CRBMMaxpoolNode( int pool_size, TENSOR_FLOAT energy_scale = 1.0f ){
            this->pool_size = pool_size;
            this->energy_scale = energy_scale;
            // load virtual function tables
            this->fp_sample       = s_sample;
            this->fp_cal_mean     = s_cal_mean;
            this->fp_feed_forward = s_feed_forward;
            this->fp_forward_bias = s_forward_bias;
            this->fp_reget_bound  = s_reget_bound;
            this->fp_reget_hidden_bound  = s_reget_hidden_bound;
            this->fp_sparse_reg   = s_sparse_reg;
        }
    };

    inline ICRBMNode *create_visible_node( const CRBMModelParam &param ){
        switch( param.model_type ){
        case model_type::BINARY_MAXPOOL    : return new CRBMBinaryNode();
        case model_type::GAUSSIAN_MAXPOOL_A: return new CRBMGaussianNode<false>( param.v_sigma );
        case model_type::GAUSSIAN_MAXPOOL_B: return new CRBMGaussianNode<true> ( param.v_sigma );
        default: return NULL;
        }
    }
    inline ICRBMNode *create_hidden_node( const CRBMModelParam &param ){
        switch( param.model_type ){
        case model_type::BINARY_MAXPOOL:     return new CRBMMaxpoolNode<false>( param.pool_size );
        case model_type::GAUSSIAN_MAXPOOL_A: return new CRBMMaxpoolNode<false>( param.pool_size );
        case model_type::GAUSSIAN_MAXPOOL_B: return new CRBMMaxpoolNode<true> ( param.pool_size, 1.0f/(param.v_sigma*param.v_sigma) );
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
        inline void refit_node_size( int input_y_max, int input_x_max ){
            if( input_y_max != v_state.y_max || input_x_max != v_state.x_max ){
                int h_y_max = input_y_max - W.y_max + 1;
                int h_x_max = input_x_max - W.x_max + 1;
                h_node->reget_hidden_bound( h_y_max, h_x_max );
                v_state.set_param( v_state.z_max, h_y_max+W.y_max-1, h_x_max+W.x_max-1 );
                h_state.set_param( h_state.z_max, h_y_max, h_x_max );
            }
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
        CRBMTrainParam param;
        bool validation_init;
        bool start_from_neg;  
        int  h_size, v_size, vv_size;
    private:
        int sample_counter, state_counter;
    private:
        TTensor4D d_W;
        TTensor1D d_h_bias, d_v_bias;
        TTensor1D h_sum_mf, h_sum_mf_grad;
    private:
        CRBMLayer layer;
        TTensor3D v_neg,  h_neg;
    private:
        inline void sync_size_to_layer( bool force_sync = false ){
            if( force_sync || layer.h_state.y_max  != h_neg.y_max || layer.h_state.x_max != h_neg.x_max ){
                start_from_neg = false;
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

            h_sum_mf      = alloc_like( d_h_bias );
            h_sum_mf_grad = alloc_like( d_h_bias );
            v_neg         = alloc_like( layer.v_state );
            h_neg         = alloc_like( layer.h_state );
            
            sample_counter = 0; state_counter = 0; 
            h_sum_mf = 0.0f; h_sum_mf_grad = 0.0f;            

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
            init_virtual_func();
            init_async();
            printf("CRBMLightTrainer initialized, h_size=%d, v_size=%d, vv_size=%d\n", h_size, v_size, vv_size );
        }
        virtual ~CRBMLightTrainer(){
            layer.free_space();
            tensor::free_space( d_h_bias );
            tensor::free_space( d_v_bias );
            tensor::free_space( d_W );
            tensor::free_space( h_sum_mf );
            tensor::free_space( h_sum_mf_grad );
            tensor::free_space( v_neg );
            tensor::free_space( h_neg );
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
            case 0 :
            default:
                {
                    h_sum_mf   = h_sum_mf * h_sum_mf_grad;
                    // leave out h_size
                    h_sum_mf  *= 2*param.sparse_lambda;                               
                    break;
                }                 
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

                // add sparse regularization and weight decay, we remember both in momentum
                if( param.use_sparse_momentum ){
                    h_bias += ( d_h_bias -= h_bias * param.wd_h + h_sum_mf * 1.0f ) * (eta * param.h_learning_rate);
				}else{
                    h_bias += ( d_h_bias -= h_bias * param.wd_h ) * (eta * param.h_learning_rate);
                    h_bias += h_sum_mf * ( - eta * param.h_learning_rate );
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
 
            W   += ( d_W -= W * param.wd_W ) * eta;                        
            d_W *= param.momentum;
        }

        // update in training
        inline void train_update(){
            TTensor3D &v_pos = layer.v_state;
            TTensor3D &h_pos = layer.h_state;

            // whether can be use persistent chain
            TTensor3D &hp = start_from_neg ? h_neg : h_pos;
			cal_cd_steps( v_pos, v_neg, h_pos, h_neg, hp, this->cd_step );
            start_from_neg = ( param.persistent_cd !=0 );

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
            layer.refit_node_size( data.y_max, data.x_max );
            this->sync_size_to_layer();
            tensor::crbm::copy_fit( layer.v_state, data );            
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
    private:        
        // do validation, return the statistics
        inline void p_validate_stats( CRBMModelStats &stats, const vector<CTensor3D> &data ){
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
                        
            for( size_t i = 0 ; i < data.size() ; i ++ ){
                setup_input( data[i] );                

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
				if( ((int)i) % param.batch_size == param.batch_size - 1 ){ 
					cal_sparse();	
					grad_sparse -= h_sum_mf;
					h_sum_mf = 0.0f; h_sum_mf_grad = 0.0f;
				}
            }                 

            h_sum_mf = 0.0f; h_sum_mf_grad = 0.0f;

            stats.h_size  = h_size;
            stats.v_size  = v_size;
            stats.vv_size = vv_size;
            stats.sample_counter += data.size();
                        
            tensor::copy( stats.grad_W, grad_W );
            tensor::copy( stats.pos_grad_h, pos_grad_h );
            tensor::copy( stats.neg_grad_h, neg_grad_h );
            tensor::copy( stats.pos_grad_v, pos_grad_v );
            tensor::copy( stats.neg_grad_v, neg_grad_v );
            tensor::copy( stats.loss, loss );
            tensor::copy( stats.grad_sparse, grad_sparse );           
        }
    private:
        inline void init_virtual_func(){
            this->fp_clone_model = s_clone_model;
            this->fp_validate_stats = s_validate_stats;
        }
    private:
        static void s_train_update( ICRBMTrainer *p_self, const CTensor3D &data ){
            CRBMLightTrainer *self = static_cast<CRBMLightTrainer*>( p_self );
            self->setup_input( data );
            self->train_update();
        }
        static void s_validate_stats( ICRBMTrainer *p_self, CRBMModelStats &stats, const vector<CTensor3D> &data ){
            CRBMLightTrainer *self = static_cast<CRBMLightTrainer*>( p_self );
            self->p_validate_stats( stats, data );
        }
        static void s_clone_model( const ICRBMTrainer *p_self, CRBMModel &md ){
            const CRBMLightTrainer *self = static_cast<const CRBMLightTrainer*>( p_self );
            tensor::copy( md.W , (self->layer).W );
            tensor::copy( md.h_bias , (self->layer).h_bias );
            tensor::copy( md.v_bias , (self->layer).v_bias );
            tensor::copy( md.d_W , self->d_W );
            tensor::copy( md.d_h_bias , self->d_h_bias );
            tensor::copy( md.d_v_bias , self->d_v_bias );            
        }     
    }; 
    
    class CRBMLightInferencer: public ICRBMInferencer{
    private:
        TTensor3D top_state;
        vector<CRBMLayer> layers;        
    public:
        CRBMLightInferencer( const CDBNModel &model, int input_y_max, int input_x_max ){
            init_tensor_engine( 0 );
            init_virtual_func();
            
            for( size_t i = 0 ; i < model.layers.size() ; i ++ ){                
                layers.push_back( CRBMLayer( model.layers[i], input_y_max, input_x_max ) );
                layers.back().reget_bound( input_y_max, input_x_max );
            }
            // top state 
            top_state.set_param( model.layers.back().param.h_max, input_y_max, input_x_max );
            tensor::alloc_space( top_state );
        }
        ~CRBMLightInferencer(){
            for( size_t i = 0 ; i < layers.size() ; i ++ )
                layers[i].free_space();
            layers.clear();

            tensor::free_space( top_state );
            destroy_tensor_engine();
        }
    private:
        inline void p_set_input( const CTensor3D &data ){
            int y_max = data.y_max;
            int x_max = data.x_max; 

            if( y_max != layers[0].v_state.y_max || 
                x_max != layers[0].v_state.x_max ){
                for( size_t i = 0; i < layers.size() ; i ++ ){
                    layers[i].refit_node_size( y_max, x_max );
                    layers[i].reget_bound( y_max, x_max );
                } 
                top_state.set_param( top_state.z_max, y_max, x_max );
            }            
            tensor::crbm::copy_fit( layers[0].v_state , data );
        }
        inline void p_get_top_bound( int &t_z_max, int &t_y_max, int &t_x_max ) const{
            t_z_max = top_state.z_max;
            t_y_max = top_state.y_max;
            t_x_max = top_state.x_max;
        } 
        inline void p_infer_top_layer( CTensor3D dout ){
            for( size_t i = 1 ; i < layers.size() ; i ++ ){
                layers[i-1].feed_forward( layers[i].v_state );
            }                       
            layers.back().feed_forward( top_state );
            tensor::copy( dout, top_state );
        }
        
    private:
        inline void init_virtual_func(){
            this->fp_set_input       = s_set_input;
            this->fp_get_top_bound   = s_get_top_bound;
            this->fp_infer_top_layer = s_infer_top_layer;
        }
        static void s_set_input( ICRBMInferencer *p_self, const CTensor3D &data ){
            static_cast<CRBMLightInferencer*>( p_self )->p_set_input( data );
        }
        static void s_get_top_bound( const ICRBMInferencer *p_self, int &t_z_max, int &t_y_max, int &t_x_max ){
            static_cast<const CRBMLightInferencer*>( p_self )->p_get_top_bound( t_z_max, t_y_max, t_x_max ); 
        }
        static void s_infer_top_layer( ICRBMInferencer *p_self, CTensor3D &dout ){
            static_cast<CRBMLightInferencer*>( p_self )->p_infer_top_layer( dout );
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

