#define _CRT_SECURE_NO_WARNINGS
/*
  simply draws the models

*/
#include "../external/CImg.h"
#include "../utils/apex_utils.h"
#include "../tensor/apex_tensor.h"
#include "../crbm/apex_crbm_model.h"

using namespace cimg_library;
using namespace apex_tensor;

const int SCALE = 2;
const int MAX_NUM_PER_LINE = 7;

inline float norm( float val, float v_min, float v_max ){
    val = (val-v_min)/(v_max-v_min);
    if( val < 0.0f ) val = 0.0f;
    if( val > 1.0f ) val = 1.0f;
    return val;
}

inline void norm_minmax( CTensor3D &m ){
	float w_max = cpu_only::max_value( m );
	float w_min = cpu_only::min_value( m );

    for( int h = 0 ; h < m.z_max ; h ++ ){
        for( int y = 0 ; y < m.y_max ; y ++ )
            for( int x = 0 ; x < m.x_max ; x ++  ){
                m[h][y][x] = norm(m[h][y][x], w_min, w_max );
            }
    }
}

inline void norm_sigmoid( CTensor3D &m, float v_bias ){
    for( int h = 0 ; h < m.z_max ; h ++ ){
        for( int y = 0 ; y < m.y_max ; y ++ )
            for( int x = 0 ; x < m.x_max ; x ++  ){
                m[h][y][x] = 1.0f / (1+expf(-m[h][y][x]- v_bias));
            }
    }
}

int main( int argc, char *argv[] ){
    if( argc < 4 ){
        printf("Usage: <model in> <image_out> <method>");        
        return 0;
    }
    apex_rbm::CDBNModel model;	
    FILE *fi = apex_utils::fopen_check( argv[1] , "rb" );
    model.load_from_file( fi );
    fclose( fi );	
    
	CTensor3D &m = model.layers[0].W[0];

    switch( atoi( argv[3] ) ){
    case 0: norm_minmax( m );  break;        
	case 2: norm_sigmoid( m, model.layers[0].v_bias[0] ); break;
    }

    // drawing procedure 
    int y_count = (m.z_max + MAX_NUM_PER_LINE-1) / MAX_NUM_PER_LINE; 
    CImg<unsigned char> img( (m.x_max+1)*MAX_NUM_PER_LINE*SCALE + 1 , (m.y_max+1)*y_count*SCALE +1 , 1 , 1 , 0 );
    
    for( int h = 0 ; h < m.z_max ; h ++ ){
        int xx = h % MAX_NUM_PER_LINE;
        int yy = h / MAX_NUM_PER_LINE;
		
        for( int y = 0 ; y < m.y_max ; y ++ )
            for( int x = 0 ; x < m.x_max ; x ++  )
                for( int dy = 0 ; dy < SCALE ; dy ++ )
					for( int dx = 0 ; dx < SCALE ; dx ++ ){
                        const int y_idx =  yy*(m.y_max+1)*SCALE + (y+1)*SCALE + dy;
						const int x_idx =  xx*(m.x_max+1)*SCALE + (x+1)*SCALE + dx;
						img( x_idx ,y_idx ) =(unsigned char) ( m[ h ][ y ][ x ]*255 );					
					} 
	}
    img.save_bmp( argv[2] );
	apex_tensor::tensor::free_space( m );
	return 0;
}
