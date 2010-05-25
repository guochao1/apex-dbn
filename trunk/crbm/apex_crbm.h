#ifndef _APEX_CRBM_H_
#define _APEX_CRBM_H_

#include "apex_crbm_model.h"
#include "apex_crbm_model_stats.h"
#include "../tensor/apex_tensor.h"

namespace apex_rbm{
    //interface of stacked rbm
    class ICRBM{
    public:        
        // feed data forward in the network to extract feature 
        virtual void feed_forward_trunk( apex_tensor::CTensor4D &dout, const apex_tensor::CTensor4D &data ) = 0;
        
        virtual void get_feed_bound( int &f_z_max, int &f_y_max, int &f_x_max ) = 0;
        
        // update the model using trunk of data 
        virtual void train_update_trunk( const apex_tensor::CTensor4D &data ) = 0;
         
        /* clone model trainied to model */
        virtual void clone_model( CDBNModel &model )const = 0;        

        /* set steps of CD */
        virtual void set_cd_step( int cd_step ) = 0;

        // validate the model by statistics
        virtual void validate_stats( CRBMModelStats &stats, const apex_tensor::CTensor4D &data ) = 0;
        
        virtual ~ICRBM(){}              
    };

    namespace factory{
        ICRBM *create_crbm( const CDBNModel &model, const CRBMTrainParam &param );
    };

};
#endif

