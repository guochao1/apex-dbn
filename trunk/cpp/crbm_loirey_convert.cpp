#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "../utils/apex_utils.h"
#include "../tensor/apex_tensor.h"
#include "../crbm/apex_crbm_model.h"

int main( int argc, char *argv[] ){
    if( argc < 4 ){
        printf("Usage: <model in> <model out> <loirey in>\n");        
        return 0;
    }
	apex_rbm::CDBNModel model;
	
    FILE *fi = apex_utils::fopen_check( argv[1] , "rb" );
    model.load_from_file( fi );
    fclose( fi );

	fi = apex_utils::fopen_check( argv[3] , "r");

	apex_rbm::CRBMModel &m = model.layers.back();
	
	m.d_h_bias = 0.0f;
	m.d_v_bias = 0.0f;
	m.d_W      = 0.0f;
	fscanf(fi,"%*d%*d%f" , &m.v_bias[0] ); 
    fscanf(fi,"%*d");
    for( int i = 0 ; i < 40 ; i ++ )
		fscanf(fi,"%f" , &m.h_bias[i] ); 

    fscanf(fi,"%*d");
    for( int i = 0 ; i < 40 ; i ++ )
        for( int y = 0 ; y < m.W.y_max ; y ++ )
            for( int x = 0 ; x < m.W.x_max ; x ++ )
                fscanf(fi,"%f", &m.W[0][i][y][x] );
    fclose( fi );
    

    FILE *fo = apex_utils::fopen_check( argv[2] , "wb" );
	model.save_to_file( fo );
    fclose( fo );
    return 0;
}
