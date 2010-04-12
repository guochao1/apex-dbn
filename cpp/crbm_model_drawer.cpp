#define _CRT_SECURE_NO_WARNINSG
/*
  simply draws the models

*/
#include "../external/CImg.h"
#include "../utils/apex_utils.h"
#include "../tensor/apex_tensor.h"
#include "../crbm/apex_crbm_model.h"

using namespace cimg_library;
using namespace apex_tensor;

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
                m[h][y][x] = norm( m[h][y][x], w_min, w_max );
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


inline void norm_minmax2( CTensor3D &m ){
    for( int h = 0 ; h < m.z_max ; h ++ ){
		float w_max = cpu_only::max_value( m[h] );
		float w_min = cpu_only::min_value( m[h] );
        for( int y = 0 ; y < m.y_max ; y ++ )
            for( int x = 0 ; x < m.x_max ; x ++  ){
                m[h][y][x] = norm( m[h][y][x], w_min, w_max );
            }
    }
}

inline void draw_mat( CTensor3D &m, const char *fname, int method, int scale, float bs = 0.0f ){
    switch( method ){
    case 0: norm_minmax( m );  break;     
	case 1: norm_minmax2( m );  break;     
    case 2: norm_sigmoid( m, bs ); break;
    }

    // drawing procedure 
    int y_count = (m.z_max + MAX_NUM_PER_LINE-1) / MAX_NUM_PER_LINE; 
    CImg<unsigned char> img( (m.x_max+1)*MAX_NUM_PER_LINE*scale + 1 , (m.y_max+1)*y_count*scale +1 , 1 , 1 , 0 );
    
    for( int h = 0 ; h < m.z_max ; h ++ ){
        int xx = h % MAX_NUM_PER_LINE;
        int yy = h / MAX_NUM_PER_LINE;
		
        for( int y = 0 ; y < m.y_max ; y ++ )
            for( int x = 0 ; x < m.x_max ; x ++  )
                for( int dy = 0 ; dy < scale ; dy ++ )
					for( int dx = 0 ; dx < scale ; dx ++ ){
                        const int y_idx =  yy*(m.y_max+1)*scale + (y+1)*scale + dy;
						const int x_idx =  xx*(m.x_max+1)*scale + (x+1)*scale + dx;
						img( x_idx ,y_idx ) =(unsigned char) ( m[ h ][ y ][ x ]*255 );					
					} 
	}
    img.save_bmp( fname );
}

inline CTensor4D pool_down( CTensor4D m , int pool_size ){
    CTensor4D mm( m.h_max, m.z_max, m.y_max*pool_size, m.x_max*pool_size );
    tensor::alloc_space( mm );
    for( int v = 0 ; v < m.h_max ; v ++ )
        for( int h = 0 ; h < m.z_max ; h ++ )
            for( int y = 0 ; y < mm.y_max ; y ++ )
                for( int x = 0 ; x < mm.x_max ; x ++ )
                    mm[v][h][y][x] = m[v][h][y/pool_size][x/pool_size] / (pool_size*pool_size);
    tensor::free_space( m ) ;
    return mm;
}   

inline CTensor4D infer_down( CTensor4D m, CTensor4D W ){
    CTensor4D mm( m.h_max, W.h_max, m.y_max +W.y_max-1 , m.x_max+W.x_max-1 );
    CTensor1D bs( W.h_max );
    tensor::alloc_space( mm );
    tensor::alloc_space( bs );
    bs = 0.0f;
  
    for( int i = 0 ; i < m.h_max ; i ++ ){
        CTensor3D mx = mm[i];
		tensor::crbm::conv2_full( mx, m[i], W, bs );
    }
    tensor::free_space( m  );
    tensor::free_space( bs );
    return mm;
}

inline void refit( CTensor4D m ){
    for( int v = 0 ; v < m.h_max ; v ++ )
        for( int h = 0 ; h < m.z_max ; h ++ )
            for( int y = 0; y < m.y_max ; y ++ )
                for( int x = 0 ; x < m.x_max ; x ++ )
                    if( m[v][h][y][x] < 0.0f )  m[v][h][y][x] = 0.0f;
}

int main( int argc, char *argv[] ){
    if( argc < 4 ){
        printf("Usage: <model in> <image_out> <method>\n");        
        return 0;
    }
    apex_rbm::CDBNModel model;	
    FILE *fi = apex_utils::fopen_check( argv[1] , "rb" );
    model.load_from_file( fi );
    fclose( fi );	

    if( model.layers.size() == 1 ){
        CTensor3D m = model.layers[0].W[0];
        draw_mat( m, argv[2], atoi( argv[3]), 2, model.layers[0].v_bias[0] );
   }else{
        int dm = 0;
        if( argc > 4 ) dm = atoi( argv[4] );

        CTensor4D &mw = model.layers.back().W;

        CTensor4D m( mw.z_max, mw.h_max, mw.y_max, mw.x_max  ); 
        tensor::alloc_space( m );

        for( int h = 0 ; h < mw.z_max ; h ++ )
            for( int v = 0 ; v < mw.h_max ; v ++ )
                for( int y = 0; y < mw.y_max ; y ++ )
                    for( int x = 0 ; x < mw.x_max ; x ++ )
                            m[h][v][y][x] = mw[v][h][y][x];
         
        for( int i = (int)model.layers.size()-2 ; i >=0 ; i -- ){
            if( dm == 0 ) refit( m );
            m = pool_down ( m, model.layers[i].param.pool_size );
            m = infer_down( m, model.layers[i].W );
        }   

        CTensor3D mm( m.h_max, m.y_max , m.x_max ); 
        mm.elem = m.elem; mm.pitch = m.pitch; 

        draw_mat( mm, argv[2], atoi( argv[3]), 2 );
        
        tensor::free_space( m );
    }
	return 0;
}
