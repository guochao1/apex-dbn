#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "../utils/apex_utils.h"
#include "../tensor/apex_tensor.h"
#include "../crbm/apex_crbm_model.h"

using namespace apex_tensor;
using namespace apex_rbm;

inline void swap( TENSOR_FLOAT &a, TENSOR_FLOAT &b ){
    TENSOR_FLOAT c = a;
    a = b; b = c;
}

inline void swap( CTensor4D &W, int a, int b ){
    for( int v = 0 ; v < W.h_max ; v ++ )
        for( int y = 0 ; y < W.y_max ; y ++ )
            for( int x = 0 ; x < W.x_max ; x ++ )
                swap( W[v][a][y][x], W[v][b][y][x] );
}    

inline void swap( CRBMModel &m, int a, int b ){
    swap( m.W  , a, b );
    swap( m.d_W, a, b );
    swap( m.h_bias[a], m.h_bias[b] );
    swap( m.d_h_bias[a], m.d_h_bias[b] );
}

int main( int argc, char *argv[] ){
    if( argc < 5 ){
        printf("Usage: <model in> <model_out> <swap a> <swap b>\n");        
        return 0;
    }
	apex_rbm::CDBNModel model;
	
    FILE *fi = apex_utils::fopen_check( argv[1] , "rb" );
    model.load_from_file( fi );
    fclose( fi );

    swap( model.layers.back() , atoi( argv[3] ), atoi(argv[4]) );
    
    FILE *fo = apex_utils::fopen_check( argv[2] , "wb" );
    model.save_to_file( fo );
    fclose( fo );

    return 0;
}

