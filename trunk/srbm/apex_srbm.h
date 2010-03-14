#ifndef _APEX_SRBM_H_
#define _APEX_SRBM_H_

#include "apex_srbm_model.h"
#include "../tensor/apex_tensor.h"

namespace apex_rbm{
    //interface of stacked rbm
    class ISRBM{
    public:
        /* update model using data */
        virtual void train_update( const apex_tensor::CTensor1D &data ) = 0;

        // update the model using trunk of data 
        virtual void train_update_trunk( const apex_tensor::CTensor2D &data ) = 0;
         
        /* clone model trainied to model */
        virtual void clone_model( SDBNModel &model )const = 0;        

        /* set steps of CD */
        virtual void set_cd_step( int cd_step ) = 0;
                
        virtual ~ISRBM(){}              
    };
    
    namespace factory{
        // create a stacked rbm
        ISRBM *create_srbm( const SDBNModel &model, const SRBMTrainParam &param );
    };

};
#endif

