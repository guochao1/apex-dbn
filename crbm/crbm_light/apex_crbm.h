#ifndef _APEX_CRBM_H_
#define _APEX_CRBM_H_

// Light version of CRBM library
#include <vector>
#include "../apex_crbm_model.h"
#include "../apex_crbm_model_stats.h"
#include "../../tensor/apex_tensor.h"

namespace apex_rbm{
    using namespace std;

    // interface training tool of a single layer of CRBM
    class ICRBMTrainer{       
    protected:
        // step of CD
        int cd_step;      
    protected:
        /* manual virtual function table, implement it for fun~ */
        // update the model using data given 
        void (*fp_train_update)   ( ICRBMTrainer *p_self, const apex_tensor::CTensor3D &data );
        // validate the model and get statistics
        void (*fp_validate_stats) ( ICRBMTrainer *p_self, CRBMModelStats &stats, const vector<apex_tensor::CTensor3D> &data );
        // clone model trainied to model 
        void (*fp_clone_model)    ( const ICRBMTrainer *p_self,CRBMModel &model );
    protected:
        // protected constructor to prevent creation
        ICRBMTrainer(){}
    public:
        // virtual destructor for save destruction
        virtual ~ICRBMTrainer(){}
    public:
        inline void set_cd_step( int cd_step ){
            this->cd_step = cd_step;
        }
    public:                
        // rejump
        inline void train_update  ( const apex_tensor::CTensor3D &data ){
            (*fp_train_update)( this, data );
        }
        inline void validate_stats( CRBMModelStats &stats, const vector<apex_tensor::CTensor3D> &data ){
            (*fp_validate_stats)( this, stats, data );
        }
        inline void clone_model   ( CRBMModel &model ){
            (*fp_clone_model)( this, model );
        }
    };

    // interface inferencer tool of stacks of CRBMs
    class ICRBMInferencer{       
    protected:
        /* virtual function table, implement it for fun~ */
        // set data in input layer
        void (*fp_set_input) ( ICRBMInferencer *p_self, const apex_tensor::CTensor3D &data ); 
        // get bound when doing feed forward, give in the input and get output bound        
        void (*fp_get_top_bound)( const ICRBMInferencer *p_self, int &t_z_max, int &t_y_max, int &t_x_max );
        // get top layer's output  
        void (*fp_infer_top_layer) ( ICRBMInferencer *p_self, apex_tensor::CTensor3D &dout );
    protected:
        // protected constructor to prevent creation of interface
        ICRBMInferencer(){}
    public:
        // virtual destructor for safe destruction
        virtual ~ICRBMInferencer(){}
    public:                
        // rejump 
        inline void set_input( const apex_tensor::CTensor3D &data ){
            (*fp_set_input)( this, data );
        } 
        inline void get_top_bound( int &t_z_max, int &t_y_max, int &t_x_max ){
            (*fp_get_top_bound)( this, t_z_max, t_y_max, t_x_max );
        }
        inline void infer_top_layer( apex_tensor::CTensor3D &dout ){
            (*fp_infer_top_layer)( this, dout);
        }
    };

    namespace factory{
        ICRBMTrainer    *create_crbm_trainer   ( const CRBMModel &model, const CRBMTrainParam &param );   
        ICRBMInferencer *create_crbm_inferencer( const CDBNModel &model, int input_y_max, int input_x_max );
    };

};
#endif

