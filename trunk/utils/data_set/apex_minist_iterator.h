#ifndef _APEX_MINIST_ITERATOR_H_
#define _APEX_MINIST_ITERATOR_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../apex_utils.h"
#include "../apex_tensor_iterator.h"

namespace apex_utils{
    /* iterator that  iterates over the MINIST data set */
    class MINISTIterator: public ITensor1DIterator{
    private:
        int idx, max_idx;
        int pitch;
        int num_image, width, height;
        int trunk_size;
        
        apex_tensor::CTensor3D data;
    private:
        char name_image_set[ 256 ];
        
        const apex_tensor::CTensor2D get_trunk( int start_idx, int max_idx ) const{
            int y_max = max_idx - start_idx;
            if( y_max > trunk_size ) y_max = trunk_size;
            
            apex_tensor::CTensor2D m( y_max, data.y_max*data.x_max );
            m.elem  = data[ start_idx ].elem;
            return m;
        } 
        
    public:    
        MINISTIterator(){
            data.elem = NULL;
            max_idx   = 1 << 30;
        }
        virtual ~MINISTIterator(){
            if( data.elem != NULL )
                apex_tensor::tensor::free_space( data );
        }
        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( name, "image_set"   ) ) strcpy( name_image_set, val );        
            if( !strcmp( name, "image_amount") ) max_idx = atoi( val );
            if( !strcmp( name, "trunk_size"  ) ) trunk_size = atoi( val );
        }

        // initialize the model
        virtual void init( void ){
            FILE *fi = apex_utils::fopen_check( name_image_set, "rb" );
            unsigned char zz[4];
            unsigned char *t_data;
            
            if( fread(zz, 4 , 1, fi ) == 0 ){
                apex_utils::error("load minist");
            }
            
            if( fread(zz, 4 , 1, fi ) == 0 ){
                apex_utils::error("load minist");
            }
            
            num_image = (int)(zz[3]) 
                | (((int)(zz[2])) << 8)
                | (((int)(zz[1])) << 16)
                | (((int)(zz[0])) << 24);
            
            if( fread(zz, 4 , 1, fi ) == 0 ){
                apex_utils::error("load minist");
            }
             
            width = (int)(zz[3]) 
                | (((int)(zz[2])) << 8)
                | (((int)(zz[1])) << 16)
                | (((int)(zz[0])) << 24);
            
            if( fread(zz, 4 , 1, fi ) == 0 ){
                apex_utils::error("load minist"); 
            }

            height = (int)(zz[3]) 
                | (((int)(zz[2])) << 8)
                | (((int)(zz[1])) << 16)
                | (((int)(zz[0])) << 24);
            
            pitch = width * height;
            t_data = new unsigned char[ num_image * pitch ];
            
            if( fread( t_data, pitch*num_image , 1 , fi) == 0 ){
                apex_utils::error("load minist");
            }        
            
            fclose( fi );
            
            data.set_param( num_image, width , height );
            apex_tensor::tensor::alloc_space( data );
            
            for( int i = 0 ; i < num_image ; i ++ )
                for( int y = 0; y < height ; y ++ )
                    for( int x = 0; x < width ; x ++ ){
						data[i][ y ][ x ] = (apex_tensor::TENSOR_FLOAT)(t_data[ i*pitch + y*width + x ] != 0 );
                    }        
            delete[] t_data;        
            idx = - trunk_size;  
            if( max_idx > data.z_max ) {
                max_idx = data.z_max;
            }
        }
        
        // move to next mat
        virtual bool next_trunk(){
            idx += trunk_size;
            if( idx >= max_idx ) return false;        
            return true;
        }
        
        // get current matrix 
        virtual const apex_tensor::CTensor2D trunk() const{
            return get_trunk( (int)idx, (int)max_idx );
        }
        
        // set before first of the item
        virtual void before_first(){
            idx = -trunk_size;
        }
        
        // trunk used for validation
        virtual const apex_tensor::CTensor2D validation_trunk()const{
            return get_trunk( (int)max_idx, num_image );
        }
    };
};
#endif
