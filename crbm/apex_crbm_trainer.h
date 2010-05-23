#ifndef _APEX_CRBM_TRAINER_H_
#define _APEX_CRBM_TRAINER_H_

#include <vector>
#include <cstdlib>
#include <cstring>

#include "apex_crbm.h"
#include "apex_crbm_model.h"
#include "../utils/apex_config.h"
#include "../utils/task/apex_tensor_update_task.h"

namespace apex_rbm{
    template<typename InputType>
	class CRBMTrainer : public apex_utils::ITensorUpdater<InputType> {
    private:
        // model parameter
        /* parameter for new layer */
        CRBMModelParam param_new_layer;
        CRBMTrainParam param_train;
        /* model of SDBN */
		CDBNModel  model;
        ICRBM<InputType>     *crbm;        
    private:
        // name of config file 
        char name_config[ 256 ];
        // 0 = new layer, 1 = continue last layer's training
        int task;
        // whether to be silent 
        int silent;
        // step of cd
        int cd_step;        
        // input bound of the image 
        int trunk_size;                
        // input model name 
        char name_model_in[ 256 ];
        // start counter of model
        int start_counter;
        // folder name of output  
        char name_model_out_folder[ 256 ];

    private:
        // name for summary and detail information of validation
        char name_summary_log[256], name_detail_log[256];
        // file for summary and detail information of validation
        FILE *fo_summary_log, *fo_detail_log;

    private:
        inline void reset_default(){
            strcpy( name_config, "crbm.conf" );
            strcpy( name_model_in, "NULL" );
            strcpy( name_model_out_folder, "models" );            
            strcpy( name_summary_log, "summary.log.txt" );           
            strcpy( name_detail_log , "detail.log.txt"  );            
            task = silent = start_counter = 0;
            cd_step = 1; trunk_size = 1;
        }
    public:
        CRBMTrainer(){
            crbm = NULL;
            reset_default();
        }
        virtual ~CRBMTrainer(){
            if( crbm != NULL ) delete crbm;            
        }

    private:
        inline void set_param_inner( const char *name, const char *val){
            if( !strcmp( name,"task"   ))            task  = atoi( val ); 
            if( !strcmp( name,"silent" ))            silent = atoi( val );        
            if( !strcmp( name,"cd_step" ))           cd_step = atoi( val );
            if( !strcmp( name,"start_counter" ))     start_counter = atoi( val );
            if( !strcmp( name,"model_in" ))          strcpy( name_model_in, val ); 
            if( !strcmp( name,"model_out_folder" ))  strcpy( name_model_out_folder, val ); 
            if( !strcmp( name,"summary_log" ))       strcpy( name_summary_log, val ); 
            if( !strcmp( name,"detail_log" ))        strcpy( name_detail_log , val ); 
            param_new_layer.set_param( name, val );
            param_train.set_param( name, val );
        }

        inline void configure(){
            apex_utils::ConfigIterator cfg( name_config );
            while( cfg.next() ){
                set_param_inner( cfg.name(), cfg.val() );
            }        
        }

        inline void load_model(){
            FILE *fi = apex_utils::fopen_check( name_model_in, "rb" );
            // load model from file 
            model.load_from_file( fi );
            fclose( fi );
        }
        
        inline void save_model(){
            char name[256];
            sprintf(name,"%s/%02d.%04d.model" , name_model_out_folder, (int)(model.layers.size()-1) , start_counter ++ );
            FILE *fo  = apex_utils::fopen_check( name, "wb" );            
            model.save_to_file( fo );
            fclose( fo );
        }            
     public:        
        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( name, "dbn_config" ) ) strcpy( name_config , val );            
            if( !strcmp( name, "trunk_size" ) ) trunk_size = atoi( val );
        }        

        virtual void init( void ){
            this->configure();
            if( strcmp( name_model_in,"NULL") != 0 ){
                 this->load_model();
			}                        
            if( task == 0 ){
                model.add_layer( param_new_layer );
            }
            crbm = factory::create_crbm<InputType>( model, param_train );
            crbm->set_cd_step( cd_step );
            // saved for further usage
            fo_summary_log = apex_utils::fopen_check( name_summary_log , "w" );
            fo_detail_log  = apex_utils::fopen_check( name_detail_log  , "w" );
        }      
        

        virtual void train_update_trunk( const InputType &data ){
            crbm->train_update_trunk( data );
        }

        virtual void validate_trunk    ( const InputType &data ){
            CRBMModelParam &param = model.layers.back().param;            
			CRBMModelStats stats( param.v_max, param.h_max, param.y_max, param.x_max );
            crbm->validate_stats( stats, data );
            stats.save_summary( fo_summary_log );
            stats.save_detail ( fo_detail_log  );
        }

        /* we end a round of training */
        virtual void round_end(){
            crbm->clone_model( model );
            this->save_model();
        }

        // do nothing 
        virtual void all_end(){
            fclose( fo_summary_log );
            fclose( fo_detail_log );
        }
    };
};

#endif
