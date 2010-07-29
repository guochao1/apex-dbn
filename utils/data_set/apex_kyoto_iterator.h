#ifndef _APEX_KYOTO_ITERATOR_H_
#define _APEX_KYOTO_ITERATOR_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../apex_utils.h"
#include "../apex_tensor_iterator.h"

namespace apex_utils{
    namespace iterator{
        template<typename T>
        inline void __kyoto_set_param( T & m, int z_max, int y_max, int x_max, unsigned int pitch );

        template<>
        inline void __kyoto_set_param<apex_tensor::CTensor1D>( apex_tensor::CTensor1D & m,  int z_max, int y_max, int x_max, unsigned int pitch ){
             m.x_max =  y_max * x_max; m.pitch = pitch*y_max; 
        }
        
        template<>
        inline void __kyoto_set_param<apex_tensor::CTensor2D>( apex_tensor::CTensor2D & m, int z_max, int y_max, int x_max, unsigned int pitch ){
             m.x_max = x_max; m.y_max = y_max*z_max; m.pitch = pitch; 
        }
        
        template<>
        inline void __kyoto_set_param<apex_tensor::CTensor3D>( apex_tensor::CTensor3D & m, int z_max, int y_max, int x_max, unsigned int pitch ){
            m.z_max = z_max; m.y_max = y_max; m.x_max = x_max; m.pitch = pitch;
        }
        
        template<typename T>
        class KyotoIterator: public IIterator<T>{
        private:
            int silent, idx;
            char name_image_set[ 256 ];            
            T dout;
            std::vector<apex_tensor::CTensor3D> data;
        public:    
            KyotoIterator(){
                silent = 0;
            }
            virtual ~KyotoIterator(){
                for( size_t i = 0 ; i < data.size() ; i ++ )
                    apex_tensor::tensor::free_space( data[i] );
                data.clear();
            }
            virtual void set_param( const char *name, const char *val ){
                if( !strcmp( name, "image_set"   ) ) strcpy( name_image_set, val );        
                if( !strcmp( name, "silent"   ) )    silent = atoi( val );
            }
            // initialize the model
            virtual void init( void ){
                int num;
                
                FILE *fi = apex_utils::fopen_check( name_image_set, "rb" );
                if( fread( &num, sizeof(int) , 1 , fi ) <= 0 ) apex_utils::error("load num image");
                
                data.resize( num );
                for( int i = 0 ; i < num ; i ++ ){
                    apex_tensor::tensor::load_from_file( data[i] , fi );
                }
                fclose( fi );
                
                if( silent == 0 )
                    printf("KyotoIterator, %d images loaded\n", (int)data.size() ); 
                before_first();
            }
            virtual void before_first(){
                idx = -1;
            }           
            virtual bool next(){
                idx ++;
                if( idx < (int)data.size() ){
                    dout.elem = data[ idx ].elem;
                    __kyoto_set_param( dout, data[idx].z_max, data[idx].y_max, data[idx].x_max, data[idx].pitch );                    
                    return true;
                } else{
                    return false;
                }
            }            
            virtual const T &value() const{
                return dout;
            }
        };
    };

    namespace deprecated{
        template<typename T>
        inline void __kyoto_set_param( T & m, int h_max, int z_max, int y_max, int x_max, unsigned int pitch );
        template<>
        inline void __kyoto_set_param<apex_tensor::CTensor2D>( apex_tensor::CTensor2D & m, int h_max, int z_max, int y_max, int x_max, unsigned int pitch ){
            m.y_max = h_max; m.x_max = z_max * y_max * x_max; m.pitch = pitch*y_max*z_max; 
        }
        
        template<>
        inline void __kyoto_set_param<apex_tensor::CTensor3D>( apex_tensor::CTensor3D & m, int h_max, int z_max, int y_max, int x_max, unsigned int pitch ){
            m.z_max = h_max; m.x_max = x_max; m.y_max = y_max*z_max; m.pitch = pitch; 
        }
        
        template<>
        inline void __kyoto_set_param<apex_tensor::CTensor4D>( apex_tensor::CTensor4D & m, int h_max, int z_max, int y_max, int x_max, unsigned int pitch ){
            m.h_max = h_max; m.z_max = z_max; m.y_max = y_max; m.x_max = x_max; m.pitch = pitch;
        }
        
        template<>
        inline void __kyoto_set_param<apex_tensor::GTensor4D>( apex_tensor::GTensor4D & m, int h_max, int z_max, int y_max, int x_max, unsigned int pitch ){
            m.h_max = h_max; m.z_max = z_max; m.y_max = y_max; m.x_max = x_max; m.pitch = pitch;
        }
        
        /* 
           iterator that  iterates over the image data set
           we extract regions from the datas to generate dataset.
        */
        template<typename T,typename StoreDataType>
        class KyotoIterator: public ITensorIterator<T>{
        private:
            int idx, max_idx, num_amount_used;
            int trunk_size;
            int width, height;
            int silent, normalize;
            int sample_gen_method, do_shuffle;
            int num_extract_per_image;        
            int repeat_round, remain_round;
            int validate_train;
            
            StoreDataType data;        
        private:
            char name_image_set[ 256 ];
        
            const T get_trunk( int start_idx, int max_idx ) const{
                int y_max = max_idx - start_idx;
                if( y_max > trunk_size ) y_max = trunk_size;            
                T m; 
                m.elem  = data[ start_idx ].elem;
                __kyoto_set_param<T>( m , y_max, data.z_max, data.y_max, data.x_max, data.pitch ); 
                return m;
            } 
            
        public:    
            KyotoIterator(){
                data.elem = NULL;
                max_idx = num_amount_used = 1 << 30;  
                silent = 0; normalize = 0; 
                sample_gen_method = 0; do_shuffle = 0;
                num_extract_per_image = 10;
                remain_round = 0; repeat_round = 1; validate_train = 0; 
            }
            virtual ~KyotoIterator(){
                if( data.elem != NULL ){
                    apex_tensor::tensor::free_space( data );
                }
            }
            virtual void set_param( const char *name, const char *val ){
                if( !strcmp( name, "image_set"   ) ) strcpy( name_image_set, val );        
                if( !strcmp( name, "image_amount") ) max_idx = atoi( val );
                if( !strcmp( name, "num_amount_used") ) num_amount_used = atoi( val );
                if( !strcmp( name, "trunk_size"  ) ) trunk_size = atoi( val );
                if( !strcmp( name, "region_width") ) width = atoi( val ); 
                if( !strcmp( name, "region_height")) height= atoi( val ); 
                if( !strcmp( name, "silent"))        silent= atoi( val ); 
                if( !strcmp( name, "normalize"))     normalize = atoi( val );
                if( !strcmp( name, "do_shuffle"))    do_shuffle = atoi( val );
                if( !strcmp( name, "repeat_round"))    repeat_round = atoi( val );
                if( !strcmp( name, "num_extract_per_image")) num_extract_per_image = atoi( val );
                if( !strcmp( name, "validate_train"))  validate_train = atoi( val );            
            }
            
            inline apex_tensor::CTensor4D gen_random_extract( const std::vector<apex_tensor::CTensor3D> &v_data ){
                apex_tensor::CTensor4D t_data( (int)v_data.size()*num_extract_per_image , v_data[0].z_max, height, width );
                apex_tensor::tensor::alloc_space( t_data );
                
                int num_used = 0;
                for( size_t i = 0 ; i < v_data.size() ; i ++ ){
                    if( v_data[i].y_max < t_data.y_max || v_data[i].x_max < t_data.x_max ) {
                        continue;
                    }
                    for( int j = 0 ; j < num_extract_per_image ; j ++ ){
                        apex_tensor::CTensor3D dd = t_data[ num_used* num_extract_per_image + j ];
                        apex_tensor::cpu_only::rand_extract( dd, v_data[i] );
                    }                        
                    num_used ++;
                }                
                
                if( num_used < (int)v_data.size() ){
                    t_data.h_max = num_used * num_extract_per_image;
                    if( silent == 0 ) printf("%d images unused, ", (int)v_data.size()- num_used );
                }
                if( silent == 0 ) printf("random extract, ");
                return t_data;
            }
            
            // initialize the model
            virtual void init( void ){
                int num;
                std::vector<apex_tensor::CTensor3D> v_data;
                
                FILE *fi = apex_utils::fopen_check( name_image_set, "rb" );
                if( fread( &num, sizeof(int) , 1 , fi ) <= 0 ) apex_utils::error("load num image");
                
                if( num >= num_amount_used ) num = num_amount_used;
                
                v_data.resize( num );
                for( int i = 0 ; i < num ; i ++ ){
                    apex_tensor::tensor::load_from_file( v_data[i] , fi );
                }
                fclose( fi );
                
                if( silent == 0 )
                    printf("Kyoto Dataset, %d images loaded,", (int)v_data.size() ); 
                
                apex_tensor::CTensor4D t_data = gen_random_extract( v_data ); 
                
                for( int i = 0 ; i < num ; i ++ )
                    apex_tensor::tensor::free_space( v_data[i] );            
            
                double var = 0.0;
                for( int i = 0 ; i < t_data.h_max ; i ++ )
                    var += apex_tensor::cpu_only::var( t_data[i] );
                if( silent == 0 ) printf("std_var=%lf, ", sqrt( var/t_data.h_max ) );
                
                if( normalize != 0 ) {
                    if( silent == 0 ) printf("normalize, ");
                    for( int i = 0 ; i < t_data.h_max ; i ++ )
                        t_data[i] += -apex_tensor::cpu_only::avg( t_data[i] );
                }
                
                if( do_shuffle != 0 ){
                    if( silent == 0 ) printf("shuffle, ");
                    apex_tensor::cpu_only::shuffle( t_data );
                }
                
                if( silent == 0 ){
                    printf( "%d sample generated, size=[%d,%d,%d]\n", t_data.h_max, t_data.z_max, t_data.y_max, t_data.x_max ); 
                }    
                if( max_idx > t_data.h_max ) max_idx = t_data.h_max;                                   
                
                data = clone( t_data );
                apex_tensor::tensor::free_space( t_data );
                
                remain_round = repeat_round;  
            }
            
            // move to next mat
            virtual bool next_trunk(){
                idx += trunk_size;
                if( idx >= max_idx ) {
                    if( -- remain_round  > 0 ){
                        idx = 0; return true;
                    }
                    else{
                        return false;        
                    }
                }
                return true;
            }
            
            // get current matrix 
            virtual const T trunk() const{
                return get_trunk( (int)idx, (int)max_idx );
            }
            
            
            // set before first of the item
            virtual void before_first(){
                idx = -trunk_size; remain_round = repeat_round;
            }
            
            // trunk used for validation
            virtual const T validation_trunk()const{
                if( validate_train ){
                // return training dataset 
                    return get_trunk( 0, data.h_max );
                }else{
                    // return validation data
                    return get_trunk( (int)max_idx, data.h_max );
                }
            }
        };
    };
};
#endif
