#ifndef _APEX_SRBM_H_
#define _APEX_SRBM_H_

#include "apex_srbm_model.h"
#include "apex_srbm_model_stats.h"
#include "../tensor/apex_tensor.h"

namespace apex_rbm{
    //interface of stacked rbm
    class ISRBM{
    public:
        /* update model using data */
        virtual void train_update( const apex_tensor::CTensor1D &data ) = 0;

		// gavinhu: update model using label and data
		virtual void train_update(const apex_tensor::CTensor1D &label, const apex_tensor::CTensor1D &data) = 0;

        // update the model using trunk of data 
        virtual void train_update_trunk( const apex_tensor::CTensor2D &data ) = 0;

		// gavinhu: update model using trunk of label and data
		virtual void train_update_trunk(const apex_tensor::CTensor2D &label, const apex_tensor::CTensor2D &data) = 0;
         
        /* clone model trainied to model */
        virtual void clone_model( SDBNModel &model )const = 0;

        /* set steps of CD */
        virtual void set_cd_step( int cd_step ) = 0;

        // validate the model by statistics
        virtual void validate_stats( SRBMModelStats &stats, const apex_tensor::CTensor2D &data ) = 0;

		// gavinhu: valid the model by statistics (with labeled data)
		virtual void validate_stats( SRBMModelStats &stats, const apex_tensor::CTensor2D &label, const apex_tensor::CTensor2D &data) = 0;

        virtual ~ISRBM(){}              
    };
    
    namespace factory{
        // create a stacked rbm
        ISRBM *create_srbm( const SDBNModel &model, const SRBMTrainParam &param );
    };

};
#endif

