#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "../utils/apex_utils.h"
#include "../tensor/apex_tensor.h"
#include "../crbm/apex_crbm_model.h"

using namespace apex_tensor;
using namespace apex_rbm;


inline void trim( CTensor4D &W, int h_max ){
    CTensor4D WW = W;
    W.z_max = h_max;
    for( int v = 0 ; v < W.h_max ; v ++ )
        for( int h = 0 ; h < W.z_max ; h ++ ){
            CTensor2D dd = W[v][h];
            apex_tensor::tensor::copy( dd , WW[v][h] );    
        }
}

inline void trim( CTensor1D &hb, int h_max ){
    hb.x_max = h_max;
}

inline void modify( CRBMModel &m, int num ){
    m.param.h_max = num;
    trim( m.W, num );
    trim( m.d_W, num );
    trim( m.h_bias, num );
    trim( m.d_h_bias, num );    
}

int main( int argc, char *argv[] ){
    if( argc < 4 ){
        printf("Usage: <model in> <model out> <num need>\n");        
        return 0;
    }
	apex_rbm::CDBNModel model;
	
    FILE *fi = apex_utils::fopen_check( argv[1] , "rb" );
    model.load_from_file( fi );
    fclose( fi );
    
    modify( model.layers.back(), atoi( argv[3] ) ); 
       
    FILE *fo = apex_utils::fopen_check( argv[2] , "wb" );
    model.save_to_file( fo );
    fclose( fo );
    return 0;
}

