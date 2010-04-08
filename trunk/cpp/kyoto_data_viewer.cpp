// this file is used to view the contents of kyoto iterator
// for testing perpuse

#define _CRT_SECURE_NO_WARNINGS


#include "../external/CImg.h"
#include "../tensor/apex_tensor.h"
#include "../utils/apex_config.h"
#include "../utils/data_set/apex_kyoto_iterator.h"

using namespace apex_utils;
using namespace cimg_library;
using namespace apex_tensor;

int MAX_NUM_PER_LINE = 7;

inline void draw_mat( const CTensor3D &m, const char *fname, int scale = 1 ){
    // drawing procedure 
    if( m.z_max < MAX_NUM_PER_LINE ) MAX_NUM_PER_LINE = m.z_max;
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

int main( int argc, char *argv[] ){
    if( argc < 2 ) {
        printf("usage:<config name>"); return 0;
    }
    apex_tensor::init_tensor_engine_cpu( 10 );
    KyotoIterator<apex_tensor::CTensor3D> itr;        
    ConfigIterator cfg( argv[1] );
    while( cfg.next() ){
        itr.set_param( cfg.name(), cfg.val() );
    }

    itr.init();
    
    itr.before_first();
    if( itr.next_trunk() ){
        draw_mat( itr.trunk(), "view_kyoto.bmp" , 1);
    }    
    return 0;
}
