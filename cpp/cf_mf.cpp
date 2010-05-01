// use matrix factorization to do MF
#include "../tensor/apex_tensor.h"

using namespace apex_tensor;

struct CFMFParam{
    float learning_rate;
    float wd, momentum;
    int   num_iter;
    int   als;
    
};
// MF W = Q P^T 
void matrix_factor( CTensor2D &hQ, CTensor2D &hP, 
                    const CTensor2DSparse &hW, 
                    CFMFParam &param ){
    TTensor2D Q,P,dQ,dP;
    TSparseIndex2D index;
    TTensor2DSparse W,E;
    Q = clone( hQ );
    P = clone( hP );    
    dQ= alloc_like( Q );
    dP= alloc_like( P );

    index = clone( hW.index );
    W = tensor::alloc_space_data( index );
    E = tensor::alloc_space_data( index );
    tensor::copy_data( W, hW );
    
    for( int i = 0; i < param.num_iter; i ++ ){
        // calculate error
        E = W - ( E = dot( Q, P.T() ));
        // update Q
        Q = Q * (1-param.wd*param.learning_rate) + ( dQ = dot( W    , P ) ) * param.learning_rate;

        if( param.als ){
            E = W - ( E = dot( Q, P.T() ));
        }
        // update P
        P = P * (1-param.wd*param.learning_rate) + ( dP = dot( W.T(), P ) ) * param.learning_rate;        
    }

    tensor::free_space( Q );
    tensor::free_space( P );
    tensor::free_space( dQ );
    tensor::free_space( dP );

    tensor::free_space_index( index );
    tensor::free_space_data ( W );
    tensor::free_space_data ( E );
}

int main( void ){
    
    return 0;
}

