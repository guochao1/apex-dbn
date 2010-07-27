#ifndef _APEX_CRBM_H_
#define _APEX_CRBM_H_

// Light version of CRBM library
#include <vector>
#include "../apex_crbm_model.h"
#include "../apex_crbm_model_stats.h"
#include "../../tensor/apex_tensor.h"
#include "../../utils/apex_utils.h"

namespace apex_rbm{
    using namespace std;
    using namespace apex_utils::iterator;

    // interface training tool of a single layer of CRBM
    class ICRBMTrainer{       
    public:
        // set steps of CD
        virtual void set_cd_step( int cd_step ) = 0;
        // update the model using data 
        virtual void train_update  ( const apex_tensor::CTensor3D& data ) = 0;
        // validation the data with 
        virtual void validate_stats( CRBMModelStats& stats, IIterator<apex_tensor::CTensor3D> *data_itr ) = 0;
        // clone the model trained to model
        virtual void clone_model( CRBMModel & model ) const = 0;
    public:
        virtual ~ICRBMTrainer(){}
    };

    // interface inferencer tool of stacks of CRBMs
    class ICRBMInferencer{       
    public:        
        // set input layer to be data 
        virtual void set_input      ( const apex_tensor::CTensor3D &data ) = 0;
        // get bound of top layer
        virtual void get_top_bound  ( int &t_z_max, int &t_y_max, int &t_x_max ) const = 0; 
        // inference top layer and store to dout
        virtual void infer_top_layer( apex_tensor::CTensor3D &dout ) = 0;    
        // forward bias to next layer
        virtual void forward_bias ( apex_tensor::CTensor1D &v_bias_next ) const = 0; 
    public:
        virtual ~ICRBMInferencer(){}
    };

    namespace factory{
        ICRBMTrainer    *create_crbm_trainer   ( const CRBMModel &model, const CRBMTrainParam &param );   
        ICRBMInferencer *create_crbm_inferencer( const CDBNModel &model, int input_y_max, int input_x_max );
    };

};
#endif

