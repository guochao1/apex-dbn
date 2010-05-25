#ifndef _APEX_CRBM_EXTRACTOR_H_
#define _APEX_CRBM_EXTRACTOR_H_

#include <vector>
#include <cstdlib>
#include <cstring>

#include "apex_crbm.h"
#include "apex_crbm_model.h"
#include "../utils/apex_config.h"
#include "../utils/task/apex_tensor_update_task.h"

namespace apex_rbm{
	class CRBMExtractor : public apex_utils::ITensorUpdater<apex_tensor::CTensor4D> {
    private:
        // model parameter
        /* parameter for new layer */
        CRBMModelParam param_new_layer;
        CRBMTrainParam param_train;
        
		CDBNModel  model;
        ICRBM     *crbm;        
    private:
        char name_model_in[256];
        char name_data_out[256];
        FILE *fo;
    private:
        int validate_count, num_count, trunk_size;
    private:
        apex_tensor::CTensor4D data_out;
    public:
        CRBMExtractor(){
            crbm = NULL; fo = NULL; validate_count = 0; num_count = 0;
            strcpy( name_data_out,"NULL");
            strcpy( name_model_in,"data_out.bin");
        }
        virtual ~CRBMExtractor(){
            if( crbm != NULL ) delete crbm;            
        }

    private:
        inline void load_model(){
            FILE *fi = apex_utils::fopen_check( name_model_in, "rb" );
            // load model from file 
            model.load_from_file( fi );
            fclose( fi );
        }
        
     public:        
        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( name, "trunk_size" ) ) trunk_size = atoi( val );
            if( !strcmp( name, "model_in")    ) strcpy( name_model_in, val );
            if( !strcmp( name, "data_out")    ) strcpy( name_data_out, val );
            if( !strcmp( name, "region_width") ) param_train.input_x_max = atoi( val ); 
            if( !strcmp( name, "region_height")) param_train.input_y_max = atoi( val ); 
        }        

        virtual void init( void ){
            this->load_model();
            
            crbm = factory::create_crbm( model, param_train );
            fo   = apex_utils::fopen_check( name_data_out, "wb" );
            fwrite( &num_count, sizeof(int), 1, fo );

            int z_max, y_max, x_max;
            crbm->get_feed_bound( z_max, y_max, x_max );
            data_out.set_param( trunk_size,z_max,y_max,x_max );
            apex_tensor::tensor::alloc_space( data_out );
        }      
        

        virtual void train_update_trunk( const apex_tensor::CTensor4D &data ){
            crbm->feed_forward_trunk( data_out, data );
            for( int i = 0 ; i < data_out.h_max ; i ++ )
                apex_tensor::tensor::save_to_file( data_out[i], fo );
            num_count += data_out.h_max;
        }

        virtual void validate_trunk( const apex_tensor::CTensor4D &data ){
            validate_count++;
            if( validate_count > 1 ) {
                apex_utils::error("we don't need multiple scan in extraction");
            } 
        }

        /* we end a round of training */
        virtual void round_end(){            
            if( validate_count == 0 ){
                // do nothing 
                fseek ( fo, 0, SEEK_SET );
                fwrite( &num_count, sizeof(int), 1, fo );
                fclose( fo );
                apex_tensor::tensor::free_space( data_out );
                printf("extracting end, %d in all, size=[%d,%d,%d]\n", num_count, data_out.z_max, data_out.y_max, data_out.x_max );
            }
        }
        virtual void all_end(){}
    };
};

#endif
