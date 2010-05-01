#ifndef _APEX_CFRBM_H_
#define _APEX_CFRBM_H_

#include "apex_cfrbm_model.h"
#include "../tensor/apex_tensor.h"
#include "../tensor/apex_tensor_sparse.h"
#include <vector>
using namespace std;
namespace apex_rbm{
    //interface of stacked rbm
    class CFSRBM{
    public:

        // update the model using trunk of data 
        virtual void train_update_trunk( const vector<apex_tensor::CSTensor2D> &data ) = 0;

		virtual void generate_model(FILE *fo) = 0;

    };
    
    namespace factory{
        // create a cf rbm
        CFSRBM *create_cfrbm( const CFSRBMModel &model, const CFSRBMTrainParam &param );
    };

};
#endif

