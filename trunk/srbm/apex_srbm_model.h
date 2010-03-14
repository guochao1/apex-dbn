#ifndef _APEX_SRBM_MODEL_H_
#define _APEX_SRBM_MODEL_H_

#include "../tensor/apex_tensor.h"
#include <vector>
#include <cstdlib>
#include <cstring>

namespace apex_rbm{

    // training parameter of srbm
    struct SRBMTrainParam{
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

        // whether to use persistent CD
        int persistent_cd;
        
        SRBMTrainParam(){
            reset_default();
        }
        /* reset the parameters to default value */
        inline void reset_default(){
            batch_size = 1;
            learning_rate     = 0.01f;
            momentum          = 0.0f;
            wd_h = wd_v = wd_W= 0.0f;
            chg_visible_bias = chg_hidden_bias = 1;
            persistent_cd = 0;
        }
        inline void set_param( const char *name, const char *val ){
            if( !strcmp("batch_size", name ) )    batch_size = atoi( val );
            if( !strcmp("learning_rate", name ) ) learning_rate = (float)atof( val );
            if( !strcmp("momentum", name ) )      momentum = (float)atof( val );
            if( !strcmp("wd_h", name ) )          wd_h = (float)atof( val );
            if( !strcmp("wd_v", name ) )          wd_v = (float)atof( val );
            if( !strcmp("wd_W", name ) )          wd_W = (float)atof( val );
            if( !strcmp("wd_node", name ) )       wd_h = wd_v = (float)atof( val );
            if( !strcmp("chg_visible_bias", name ) ) chg_visible_bias = atoi( val );
            if( !strcmp("chg_hidden_bias", name ) )  chg_hidden_bias  = atoi( val );
            if( !strcmp("persistent_cd", name ) )    persistent_cd  = atoi( val );
        }
    };
    
    // model paramter for each layer
    struct SRBMModelParam{        
        /* 
           type of the model 
           reserved for constructor and factory 
         */
        int model_type;
        
        /* number of visible and hidden unit */
        int v_max , h_max;

        /* the gaussian variance in gaussian unit */
        float v_sigma, h_sigma;
        
        /* the mean of bias in prior distribution */
        float v_bias_prior_mean, h_bias_prior_mean;        
        
        // standard variance for initialization
        float v_init_sigma, h_init_sigma, w_init_sigma;

        inline void set_param( const char *name, const char *val ){
            if( !strcmp("model_type", name ))     model_type = atoi( val );
            if( !strcmp("v_max", name ) )         v_max = atoi( val );
            if( !strcmp("h_max", name ) )         h_max = atoi( val );
            if( !strcmp("v_sigma", name ) )       v_sigma = (float)atof( val );
            if( !strcmp("v_init_sigma", name ) )  v_init_sigma = (float)atof( val );
            if( !strcmp("h_init_sigma", name ) )  h_init_sigma = (float)atof( val );
            if( !strcmp("w_init_sigma", name ) )  w_init_sigma = (float)atof( val );
            if( !strcmp("init_sigma", name ) )    v_init_sigma = h_init_sigma = w_init_sigma = (float)atof( val );
            if( !strcmp("h_bias_prior_mean", name ) ) h_bias_prior_mean = (float)atof( val );
            if( !strcmp("v_bias_prior_mean", name ) ) v_bias_prior_mean = (float)atof( val );            
        }
    };

    // model of srbm
    struct SRBMModel{
        SRBMModelParam param;
        // node bias
        apex_tensor::CTensor1D h_bias, v_bias;
        apex_tensor::CTensor2D Wvh;

        // change of weight
        apex_tensor::CTensor1D d_h_bias, d_v_bias;
        apex_tensor::CTensor2D d_Wvh;
        
        inline void alloc_space(){
            h_bias.set_param( param.h_max );
            v_bias.set_param( param.v_max );
            d_h_bias.set_param( param.h_max );
            d_v_bias.set_param( param.h_max );
            Wvh.set_param  ( param.v_max, param.h_max );
            d_Wvh.set_param( param.v_max, param.h_max );

            apex_tensor::tensor::alloc_space( h_bias );
            apex_tensor::tensor::alloc_space( v_bias );
            apex_tensor::tensor::alloc_space( d_h_bias );
            apex_tensor::tensor::alloc_space( d_h_bias );
            apex_tensor::tensor::alloc_space( Wvh );
            apex_tensor::tensor::alloc_space( d_Wvh );
        }

        inline void free_space(){
            apex_tensor::tensor::free_space( h_bias );
            apex_tensor::tensor::free_space( v_bias );
            apex_tensor::tensor::free_space( d_h_bias );
            apex_tensor::tensor::free_space( d_h_bias );
            apex_tensor::tensor::free_space( Wvh );
            apex_tensor::tensor::free_space( d_Wvh );
        }

        inline void load_from_file( FILE *fi ){            
            if( fread( &param, sizeof(SRBMModelParam) , 1 , fi ) == 0 ){
                printf("error loading srbm model\n"); exit( -1 );
            }
            alloc_space();
            apex_tensor::tensor::load_from_file( h_bias, fi );
            apex_tensor::tensor::load_from_file( v_bias, fi );
            apex_tensor::tensor::load_from_file( Wvh, fi );
            apex_tensor::tensor::load_from_file( d_h_bias, fi );
            apex_tensor::tensor::load_from_file( d_v_bias, fi );
            apex_tensor::tensor::load_from_file( d_Wvh, fi );
        }
        
        inline void save_to_file( FILE *fo ) const{
            fwrite( &param, sizeof(SRBMModelParam) , 1 , fo );
            apex_tensor::tensor::save_to_file( h_bias, fo );
            apex_tensor::tensor::save_to_file( v_bias, fo );
            apex_tensor::tensor::save_to_file( Wvh, fo );
            apex_tensor::tensor::save_to_file( d_h_bias, fo );
            apex_tensor::tensor::save_to_file( d_v_bias, fo );
            apex_tensor::tensor::save_to_file( d_Wvh, fo );
        }
                        
        inline void rand_init(){
            apex_tensor::tensor::sample_gaussian( h_bias, param.h_init_sigma );
            apex_tensor::tensor::sample_gaussian( v_bias, param.v_init_sigma );
            apex_tensor::tensor::sample_gaussian( Wvh   , param.w_init_sigma );
            h_bias += param.h_bias_prior_mean;
            v_bias += param.v_bias_prior_mean;
            
            d_Wvh    = 0.0f;
            d_h_bias = 0.0f;
            d_v_bias = 0.0f;
        }       

        inline void save_to_text( FILE *fo )const{
            fprintf( fo , "v_max=%d,h_max=%d\n", param.v_max, param.h_max );

            fprintf( fo , "v_bias:\n" );
            for( int v = 0 ; v < param.v_max ; v ++ )
                fprintf( fo, "%f\t", (float)v_bias[v] );

            fprintf( fo , "\nh_bias:\n" );
            for( int h = 0 ; h < param.h_max ; h ++ )
                fprintf( fo, "%f\t", (float)h_bias[h] );

            for( int v = 0 ; v < param.v_max ; v ++ ){
                for( int h = 0 ; h < param.h_max ; h ++ ){                    
                    fprintf(fo,"%f\t", (float)Wvh[v][h] );                     
                }
                fprintf( fo, "\n" );
            }
        }
    };
    
    struct SDBNModel{
        std::vector<SRBMModel> layers;
        ~SDBNModel(){
            for( size_t i = 0 ; i < layers.size() ; i ++ ){
                layers[i].free_space();
            }
        }
        
        inline void load_from_file( FILE *fi ){
            layers.clear();
            size_t count, s;

            s = fread( &count, sizeof(size_t),  1 , fi );
            if( s <=0 ){
                fprintf(stderr,"error loading SDBN model\n"); exit( -1 );
            }

            layers.resize( count );
            for( size_t i = 0 ; i < count ; i ++ ){                
                layers[i].load_from_file( fi );
            }            
        }

        inline void save_to_file( FILE * fo )const{
            size_t count = layers.size();
            fwrite( &count, sizeof(size_t) , 1, fo );
            for( size_t i = 0 ; i < count ; i ++ ){
                layers[ i ].save_to_file( fo );
            }            
        }
        
        inline void save_to_text( FILE *fo ) const{
            fprintf( fo, "%d layers\n\n", layers.size());
            for( size_t i = 0 ; i < layers.size() ; i ++ ){
                fprintf( fo,"layer[%d]:\n", i );
                layers[i].save_to_text( fo );
            }
        }
        // add a layer to the dbn model
        inline void add_layer( const SRBMModelParam  &p ){
            layers.resize( layers.size() + 1 );
            layers.back().param = p;
            layers.back().alloc_space();            
        }                        
    };    
};
#endif