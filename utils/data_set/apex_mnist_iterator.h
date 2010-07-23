#ifndef _APEX_MNIST_ITERATOR_H_
#define _APEX_MNIST_ITERATOR_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../apex_utils.h"
#include "../apex_tensor_iterator.h"

namespace apex_utils{
    template<typename T>
    inline void __mnist_set_param( T & m, int z_max, int y_max, int x_max, unsigned int pitch );
    
    template<>
    inline void __mnist_set_param<apex_tensor::CTensor2D>( apex_tensor::CTensor2D & m, int z_max, int y_max, int x_max, unsigned int pitch ){
        m.y_max = z_max; m.x_max = y_max * x_max; m.pitch = pitch * y_max;
    }
    template<>
    inline void __mnist_set_param<apex_tensor::CTensor3D>( apex_tensor::CTensor3D & m, int z_max, int y_max, int x_max, unsigned int pitch ){
        m.z_max = z_max; m.y_max = y_max; m.x_max = x_max; m.pitch = pitch;
    }
    template<>
    inline void __mnist_set_param<apex_tensor::CTensor4D>( apex_tensor::CTensor4D & m, int z_max, int y_max, int x_max, unsigned int pitch ){
        m.h_max = z_max; m.z_max = 1; m.y_max = y_max; m.x_max = x_max; m.pitch = pitch; 
    }

    /* iterator that  iterates over the MNIST data set */
    template<typename T>
    class MNISTIterator: public ITensorIterator<T>{
    private:
        int idx, max_idx;
        int pitch;
        int num_image, width, height;
        int trunk_size;
        
        apex_tensor::CTensor3D data;
    private:
        char name_image_set[ 256 ];
        
        const T get_trunk( int start_idx, int max_idx ) const{
            int y_max = max_idx - start_idx;
            if( y_max > trunk_size ) y_max = trunk_size;            
            T m; 
            m.elem  = data[ start_idx ].elem;
            __mnist_set_param<T>( m , y_max, data.y_max, data.x_max, data.pitch ); 
            return m;
        } 
        
    public:    
        MNISTIterator(){
            data.elem = NULL;
            max_idx   = 1 << 30;
        }
        virtual ~MNISTIterator(){
            if( data.elem != NULL )
                delete [] data.elem;
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
                apex_utils::error("load mnist");
            }
            
            if( fread(zz, 4 , 1, fi ) == 0 ){
                apex_utils::error("load mnist");
            }
            
            num_image = (int)(zz[3]) 
                | (((int)(zz[2])) << 8)
                | (((int)(zz[1])) << 16)
                | (((int)(zz[0])) << 24);
            
            if( fread(zz, 4 , 1, fi ) == 0 ){
                apex_utils::error("load mnist");
            }
             
            width = (int)(zz[3]) 
                | (((int)(zz[2])) << 8)
                | (((int)(zz[1])) << 16)
                | (((int)(zz[0])) << 24);
            
            if( fread(zz, 4 , 1, fi ) == 0 ){
                apex_utils::error("load mnist"); 
            }

            height = (int)(zz[3]) 
                | (((int)(zz[2])) << 8)
                | (((int)(zz[1])) << 16)
                | (((int)(zz[0])) << 24);
            
            pitch = width * height;
            t_data = new unsigned char[ num_image * pitch ];
            
            if( fread( t_data, pitch*num_image , 1 , fi) == 0 ){
                apex_utils::error("load mnist");
            }        
            
            fclose( fi );
            
            data.set_param( num_image, height , width );
			data.pitch = height * sizeof(apex_tensor::TENSOR_FLOAT);
            data.elem  = new apex_tensor::TENSOR_FLOAT[ num_image*height*width ];
            
            for( int i = 0 ; i < num_image ; i ++ )
                for( int y = 0; y < height ; y ++ )
                    for( int x = 0; x < width ; x ++ ){
						data[ i ][ y ][ x ] = (apex_tensor::TENSOR_FLOAT)(t_data[ i*pitch + y*width + x ]) /255.0f ;
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
        virtual const T trunk() const{
            return get_trunk( (int)idx, (int)max_idx );
        }
        

        // set before first of the item
        virtual void before_first(){
            idx = -trunk_size;
        }
        
        // trunk used for validation
        virtual const T validation_trunk()const{
            return get_trunk( (int)max_idx, num_image );
        }
    };
};
#endif

