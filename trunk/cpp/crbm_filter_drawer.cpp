#define _CRT_SECURE_NO_WARNINSG
/*
  simply draws the models by filtering the image of certain dataset using the model
*/
#include "../external/CImg.h"
#include "../utils/apex_utils.h"
#include "../tensor/apex_tensor.h"
#include "../crbm/apex_crbm_model.h"
#include "../utils/apex_config.h"
#include "../utils/data_set/apex_kyoto_iterator.h"

using namespace cimg_library;
using namespace apex_tensor;
using namespace apex_utils;

const int MAX_NUM_PER_LINEX = 7;

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

inline void draw_mat( const CTensor3D &m, const char *fname, int scale ){
    int MAX_NUM_PER_LINE = MAX_NUM_PER_LINEX;
    if( m.z_max < MAX_NUM_PER_LINEX ) MAX_NUM_PER_LINE = m.z_max;

    // drawing procedure 
    int y_count = (m.z_max+ MAX_NUM_PER_LINE-1) / MAX_NUM_PER_LINE; 
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

inline void draw_mat( CTensor2D m, const char *fname, int scale ){
    CTensor3D mm(1,m.y_max,m.x_max);
    mm.pitch = m.pitch;
    mm.elem = m.elem;
    draw_mat( mm, fname, scale );
}

inline CTensor3D infer_up( CTensor3D &v, const apex_rbm::CRBMModel &model ){
    CTensor3D h( model.param.h_max, v.y_max - model.param.y_max + 1,  v.x_max - model.param.x_max + 1 );
    tensor::alloc_space( h );
    tensor::crbm::conv2_r_valid( h, v, model.W, model.h_bias );
    tensor::crbm::norm_maxpooling_2D( h, h, model.param.pool_size );
    tensor::free_space( v );
    return h;    
}

inline CTensor3D pool_up( CTensor3D &h, const apex_rbm::CRBMModel &model, int vv_num ){
    CTensor3D p( vv_num, h.y_max/model.param.pool_size, h.x_max / model.param.pool_size );
    tensor::alloc_space( p );
    tensor::crbm::pool_up( p, h, model.param.pool_size );
    tensor::free_space( h );
    return p;    
}

inline void filter_draw( const CTensor2D &pic,  const apex_rbm::CDBNModel &model, int idx, int mm, int s_max ){
    CTensor3D v_data( 1 , pic.y_max, pic.x_max );

    tensor::alloc_space( v_data );
    CTensor2D vc = v_data[0];
    tensor::copy( vc , pic );
    
    for( size_t i = 0 ; i < model.layers.size() ; i ++ ){
        v_data = infer_up( v_data, model.layers[i] );
        if( i + 1 < model.layers.size() ){
            v_data = pool_up( v_data, model.layers[i], model.layers[i+1].param.v_max );
        }
    }        

    switch( mm ){
    case 0: norm_minmax( v_data );  break;
    case 1: norm_minmax2( v_data );  break;
    }
    
    printf("infer end, start drawing\n");
    
    
    for( int i = 0 ; i < v_data.z_max; i ++ ) { 
        char fname[256];
        sprintf( fname, "kyoto.filter.%03d.%02d.bmp", i, idx );
        draw_mat( v_data[i], fname , 1 );
        if( i > 0 && i < s_max ) v_data[0] += v_data[i];
    }
    if( s_max > 0 ){
        norm_minmax2( v_data );
        
        char fname[256];
        sprintf( fname, "kyoto.filter.all.%02d.bmp",  idx );
        draw_mat( v_data[0], fname , 1 );
    }
    tensor::free_space( v_data );
}

int main( int argc, char *argv[] ){
    if( argc < 4 ){
        printf("Usage: <config> <model in> <method>\n");        
        return 0;
    }
    int s_max = 0;
    if( argc > 4 ) s_max = atoi( argv[4] );

    apex_rbm::CDBNModel model;	
    FILE *fi = apex_utils::fopen_check( argv[2] , "rb" );
    model.load_from_file( fi );
    fclose( fi );	

    apex_tensor::init_tensor_engine_cpu( 10 );
    KyotoIterator<apex_tensor::CTensor3D,apex_tensor::CTensor4D> itr;        
    ConfigIterator cfg( argv[1] );
    while( cfg.next() ){
        itr.set_param( cfg.name(), cfg.val() );
    }

    itr.init(); 
    itr.before_first();
    if( itr.next_trunk() ){
        draw_mat( itr.trunk(), "view_kyoto.bmp" , 1);
        const CTensor3D &tk = itr.trunk();
        for( int i = 0 ; i < tk.z_max ; i ++ ){            
            filter_draw( tk[i] , model, i, atoi(argv[3]), s_max ); 
        }
    }                
	return 0;
}
