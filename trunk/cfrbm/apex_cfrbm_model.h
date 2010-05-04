#ifndef _APEX_CFSRBM_MODEL_H_
#define _APEX_CFSRBM_MODEL_H_

#include "../tensor/apex_tensor.h"
#include "../tensor/apex_tensor_sparse.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

using namespace std;
namespace apex_rbm{

    // training parameter of srbm
    struct CFSRBMTrainParam{
        /* batch_size */
        int batch_size;
        
        /* learning rate */
        float learning_rate;

        /* momentum */
        float momentum;

        /* weight decay of h, v, W , sum of W*/
        float wd_h, wd_v, wd_W;

        // decay of learning rate 
        float learning_rate_decay;

        // whether do we fix the bias of the model
        int chg_visible_bias, chg_hidden_bias;        

        // the number of CD steps
        int cd_step;
        
        CFSRBMTrainParam(){
            reset_default();
        }

        /* reset the parameters to default value */
        inline void reset_default(){
            batch_size = 1;
            learning_rate     = 0.01f;
            momentum          = 0.0f;
            wd_h = wd_v = wd_W= 0.0f;
            chg_visible_bias = chg_hidden_bias = 1;
            cd_step = 1;
        }
        inline void set_param( const char *name, const char *val ){
            if( !strcmp("batch_size", name ) )    batch_size = atoi( val );
            if( !strcmp("learning_rate", name ) ) learning_rate = (float)atof( val );
            if( !strcmp("momentum", name ) )      momentum = (float)atof( val );
            if( !strcmp("wd_h", name ) )          wd_h = (float)atof( val );
            if( !strcmp("wd_v", name ) )          wd_v = (float)atof( val );
            if( !strcmp("wd_W", name ) )          wd_W = (float)atof( val );
            if( !strcmp("chg_visible_bias", name ) ) chg_visible_bias = atoi( val );
            if( !strcmp("chg_hidden_bias", name ) )  chg_hidden_bias  = atoi( val );
            if( !strcmp("cd_step", name ) )    cd_step = atoi( val );
        }
    };

    // model paramter for each layer
    struct CFSRBMModelParam{        
        /* 
           type of the model 
           reserved for constructor and factory 
         */
        int softmax_size;
        
        /* number of visible and hidden unit */
        int v_max , h_max;

		/* initialize factor of the v and h */
		int v_init_sigma, h_init_sigma, w_init_sigma;

         CFSRBMModelParam(){
            reset_default();
        }

        /* reset the parameters to default value */
        inline void reset_default(){
			this->softmax_size = -1;
			this->v_max = -1;
			this->h_max = -1;
			this->v_init_sigma = -1;
			this->h_init_sigma = -1;
			this->w_init_sigma = -1;
        }
       
        inline void set_param( const char *name, const char *val ){

            if( !strcmp("size_softmax", name ))     softmax_size = atoi( val );
            if( !strcmp("v_max", name ) )         v_max = atoi( val );
            if( !strcmp("h_max", name ) )         h_max = atoi( val );
            if( !strcmp("v_init_sigma", name ) )         v_init_sigma = atoi( val );
            if( !strcmp("h_init_sigma", name ) )         h_init_sigma = atoi( val );
            if( !strcmp("w_init_sigma", name ) )         w_init_sigma = atoi( val );

        }
    };

    // model of srbm
    struct CFSRBMModel{
        CFSRBMModelParam param;
        // node bias
        apex_tensor::CTensor1D h_bias;
		apex_tensor::CTensor2D v_bias;
        apex_tensor::CTensor3D W;

        apex_tensor::CTensor1D d_h_bias;
		apex_tensor::CTensor2D d_v_bias;
        apex_tensor::CTensor3D d_W;

		CFSRBMModel( CFSRBMModelParam param ){
			this->param = param;
			alloc_space();
			rand_init();
		}
        
        inline void alloc_space(){
            h_bias.set_param( param.h_max );
            v_bias.set_param(param.softmax_size, param.v_max );
            W.set_param  ( param.softmax_size, param.v_max, param.h_max );

		   	d_h_bias.set_param( param.h_max );
            d_v_bias.set_param(param.softmax_size, param.v_max );
            d_W.set_param  ( param.softmax_size, param.v_max, param.h_max );

            apex_tensor::tensor::alloc_space( h_bias );
            apex_tensor::tensor::alloc_space( v_bias );
            apex_tensor::tensor::alloc_space( W );
			apex_tensor::tensor::alloc_space( d_h_bias );
            apex_tensor::tensor::alloc_space( d_v_bias );
            apex_tensor::tensor::alloc_space( d_W );

        }

        inline void free_space(){
            apex_tensor::tensor::free_space( h_bias );
            apex_tensor::tensor::free_space( v_bias );
            apex_tensor::tensor::free_space( W );
			apex_tensor::tensor::free_space( d_h_bias );
            apex_tensor::tensor::free_space( d_v_bias );
            apex_tensor::tensor::free_space( d_W );

        }
                                
        inline void rand_init(){
            apex_tensor::tensor::sample_gaussian( h_bias, param.h_init_sigma );
            apex_tensor::tensor::sample_gaussian (v_bias, param.v_init_sigma );
            apex_tensor::tensor::sample_gaussian( W   , param.w_init_sigma );
            
            d_W      = 0.0f;
            d_h_bias = 0.0f;
            d_v_bias = 0.0f;
        }       
    };

};
#endif
