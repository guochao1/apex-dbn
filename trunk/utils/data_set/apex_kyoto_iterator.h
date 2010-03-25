#ifndef _APEX_KYOTO_ITERATOR_H_
#define _APEX_KYOTO_ITERATOR_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../apex_utils.h"
#include "../apex_tensor_iterator.h"

namespace apex_utils{
    template<typename T>
    inline void __kyoto_set_param( T & m, int z_max, int y_max, int x_max  );
    
    template<>
    inline void __kyoto_set_param<apex_tensor::CTensor2D>( apex_tensor::CTensor2D & m, int z_max, int y_max, int x_max  ){
        m.y_max = z_max; m.x_max = y_max * x_max; 
    }
    template<>
    inline void __kyoto_set_param<apex_tensor::CTensor3D>( apex_tensor::CTensor3D & m, int z_max, int y_max, int x_max  ){
        m.z_max = z_max; m.y_max = y_max; m.x_max = x_max; 
    }
    template<>
    inline void __kyoto_set_param<apex_tensor::CTensor4D>( apex_tensor::CTensor4D & m, int z_max, int y_max, int x_max  ){
        m.h_max = z_max; m.z_max = 1; m.y_max = y_max; m.x_max = x_max; 
    }

    /* iterator that  iterates over the MINIST data set */
    template<typename T>
    class KyotoIterator: public ITensorIterator<T>{
    private:
        int idx, max_idx;
        int trunk_size;
        int width, height;
        int silent, normalize;
        apex_tensor::CTensor3D data;
    private:
        char name_image_set[ 256 ];
        
        const T get_trunk( int start_idx, int max_idx ) const{
            int y_max = max_idx - start_idx;
            if( y_max > trunk_size ) y_max = trunk_size;            
            T m; 
            m.pitch = data.pitch;
            m.elem  = data[ start_idx ].elem;
            __kyoto_set_param<T>( m , y_max, data.y_max, data.x_max ); 
            return m;
        } 
        
    public:    
        KyotoIterator(){
            data.elem = NULL;
            max_idx   = 1 << 30;
            silent = 0; normalize = 0;
        }
        virtual ~KyotoIterator(){
            if( data.elem != NULL )
                delete [] data.elem;
        }
        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( name, "image_set"   ) ) strcpy( name_image_set, val );        
            if( !strcmp( name, "image_amount") ) max_idx = atoi( val );
            if( !strcmp( name, "trunk_size"  ) ) trunk_size = atoi( val );
            if( !strcmp( name, "region_width") ) width = atoi( val ); 
            if( !strcmp( name, "region_height")) height= atoi( val ); 
            if( !strcmp( name, "silent"))        silent= atoi( val ); 
            if( !strcmp( name, "normalize"))     normalize = atoi( val );
        }

        // initialize the model
        virtual void init( void ){
            apex_tensor::CTensor3D tdata;
            FILE *fi = apex_utils::fopen_check( name_image_set, "rb" );
            apex_tensor::tensor::load_from_file( tdata , fi );
            fclose( fi );

            int yy_max = tdata.y_max / height;
            int xx_max = tdata.x_max / width;
            data.set_param( tdata.z_max*yy_max*xx_max, height, width );
            apex_tensor::tensor::alloc_space( data );

            for( int i = 0 ; i < tdata.z_max ; i ++ ){
                for( int y = 0 ; y < yy_max ; y ++ )
                    for( int x = 0 ; x < xx_max ; x ++ ){
						const int yy = y * height;
						const int xx = x * width;
						apex_tensor::CTensor2D &dd = data[ i*yy_max*xx_max + y*xx_max + x ];
                        for( int dy = 0 ; dy < height ; dy ++ )
                            for( int dx = 0 ; dx < width ; dx ++ )
                                dd[dy][dx] 
                                    = tdata[i][yy+dy][xx+dx];

                        // normalize to standard distribution
                        if( normalize != 0 ) {
                            dd += -apex_tensor::cpu_only::avg( dd );
                        }
                    }                        
            }                

            apex_tensor::tensor::free_space( tdata );

            if( max_idx > data.z_max ) max_idx = data.z_max;
            if( silent == 0 )
                printf("Kyoto Dataset, %d images loaded, %d sample generated\n", tdata.z_max, data.z_max );            
        }
        
        // move to next mat
        virtual bool next_trunk(){
            idx += trunk_size;
            if( idx >= max_idx ) return false;        
            return true;
        }
        
        // get current matrix 
        virtual const T trunk() const{
            return get_trunk( (int)idx, (int)max_idx );
        }
        

        // set before first of the item
        virtual void before_first(){
            idx = -trunk_size;
        }
        
        // trunk used for validation
        virtual const T validation_trunk()const{
            return get_trunk( (int)max_idx, data.z_max );
        }
    };
};
#endif
