#ifndef _APEX_CRBM_CPP_
#define _APEX_CRBM_CPP_

#include "apex_crbm.h"
#include "apex_crbm_model.h"
#include "apex_crbm_model_stats.h"
#include "../tensor/apex_tensor.h"

#include <vector>

namespace apex_rbm{
    using namespace std;
    using namespace apex_tensor;
    
    namespace factory{
        // create a stacked rbm
        ICRBM *create_crbm( const CDBNModel &model, const CRBMTrainParam &param ){
            return  NULL;
        }
    };
};

#endif

