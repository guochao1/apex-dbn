#ifndef _APEX_CRBM_H_
#define _APEX_CRBM_H_

#include "apex_crbm_model.h"
#include "apex_crbm_model_stats.h"
#include "../tensor/apex_tensor.h"

namespace apex_rbm{
    //interface of stacked rbm
    template<typename InputType>
    class ICRBM{
    public:
        // update the model using trunk of data 
        virtual void train_update_trunk( const InputType &data ) = 0;
         
        /* clone model trainied to model */
        virtual void clone_model( CDBNModel &model )const = 0;        

        /* set steps of CD */
        virtual void set_cd_step( int cd_step ) = 0;

        // validate the model by statistics
        virtual void validate_stats( CRBMModelStats &stats, const InputType &data ) = 0;
        
        virtual ~ICRBM(){}              
    };

    namespace factory{
        template<typename InputType>
        ICRBM<InputType> * create_crbm( const CDBNModel &model, const CRBMTrainParam &param );
    };

};
#endif

