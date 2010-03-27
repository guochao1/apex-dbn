#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "../utils/apex_utils.h"
#include "../tensor/apex_tensor.h"
#include "../crbm/apex_crbm_model.h"

int main( int argc, char *argv[] ){
    if( argc < 3 ){
        printf("Usage: <model in> <text_out>");        
        return 0;
    }
	apex_rbm::CDBNModel model;
	
    FILE *fi = apex_utils::fopen_check( argv[1] , "rb" );
    model.load_from_file( fi );
    fclose( fi );

    FILE *fo = apex_utils::fopen_check( argv[2] , "w" );
    model.save_to_text( fo );
    fclose( fo );
    return 0;
}

