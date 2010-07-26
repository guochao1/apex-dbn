#ifndef _APEX_CRBM_TRAIN_H_
#define _APEX_CRBM_TRAIN_H_

#include <vector>

#include "apex_crbm.h"
#include "../../utils/apex_utils.h"
#include "../../tensor/apex_tensor.h"

namespace apex_rbm{
    using namespace std;
    using namespace apex_tensor;
    using namespace apex_utils::iterator;

    // preserve
    class CRBMInferIterator: public IIterator<CTensor3D>{
    private:
        CTensor3D tmp_data;
        IIterator<CTensor3D> *base_itr;
        ICRBMInferencer *infer;
    private:
        inline void sync_size(){
            int z_max, y_max, x_max;
            infer->get_top_bound( z_max, y_max, x_max );
            tmp_data.set_param( z_max, y_max, x_max );
        }
    public:
        CRBMInferIterator( IIterator<CTensor3D> *base_itr, 
                           ICRBMInferencer *infer ){
            this->base_itr = base_itr;
            this->infer    = infer;
            sync_size();
            tensor::alloc_space( tmp_data );
        }

        virtual ~CRBMInferIterator(){
            if( tmp_data.elem != NULL ){
                tensor::free_space( tmp_data );               
                tmp_data.elem = NULL; 
            }
            if( base_itr != NULL ){
                delete base_itr;
                base_itr = NULL;
            }
            if( infer != NULL ){
                delete infer; 
                infer = NULL;
            }
        }

        virtual void before_first(){
            base_itr->before_first();
        }
        virtual bool next(){
            if( base_itr->next() ){
                infer->set_input( base_itr->value() );
                sync_size();
                infer->infer_top_layer( tmp_data );
                return true;
            }else{
                return false;
            }
        }
        virtual const CTensor3D &value() const{
            return tmp_data;
        }
        virtual void set_param( const char *name, const char *val ){
            base_itr->set_param( name, val );
        }
        virtual void init(){
            base_itr->init();
        }
    };
    
    // sample data out of previous data 
    class Tensor3DSampleIterator: IIterator<CTensor3D>{
    private:        
        int sample_freq;
        int sample_y_max, sample_x_max;
        CTensor3D tmp_data;
        IIterator<CTensor3D> *base_itr;
        int sample_counter;
    public:
        Tensor3DSampleIterator(){ 
            base_itr = NULL; tmp_data.elem = NULL; 
        }
        Tensor3DSampleIterator( IIterator<CTensor3D> *base_itr ){
            this->base_itr = base_itr;
            tmp_data.elem  = NULL;
        }
        virtual ~Tensor3DSampleIterator(){
            if( base_itr != NULL ) {
                delete base_itr; base_itr = NULL;
            }
            if( tmp_data.elem != NULL ){
                tensor::free_space( tmp_data );
            }
        }
                
        inline void set_base_itr( IIterator<CTensor3D> *base_itr ){
            this->base_itr = base_itr;
        }

        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( name, "sample_y_max") ) sample_y_max = atoi( val ); 
            if( !strcmp( name, "sample_x_max") ) sample_y_max = atoi( val ); 
            base_itr->set_param( name, val );
        }
        
        virtual void init( void ){
            if( base_itr != NULL ) 
                apex_utils::error("no base iterator provided");
            base_itr->before_first();

            if( base_itr->next() ){
                tmp_data.set_param( base_itr->value().z_max, sample_y_max, sample_x_max ); 
                tensor::alloc_space( tmp_data );
            }else{
                apex_utils::error("empty base iterator");
            }
            before_first();
        }

        virtual void before_first(){
            base_itr->before_first();
            sample_counter = 0;
        }
        virtual bool next(){
            if( sample_counter <= 0 ){
                while( base_itr->next() ){
                    if( base_itr->value().y_max >= sample_y_max &&
                        base_itr->value().x_max >= sample_x_max ){
                        sample_counter = sample_freq;
                        break;
                    }
                }
            }
            if( sample_counter > 0 ){
                sample_counter --;
                cpu_only::rand_extract( tmp_data, base_itr->value() );
                return true;
            }else {
                return false;            
            }
        }
        virtual const CTensor3D &value() const{
            return tmp_data;
        }
    };
    
    // iterator that buffers results of previous iterator and do some processing   
    class Tensor3DBufferIterator: IIterator<CTensor3D>{
    private:
        int idx;
        int silent,do_shuffle, max_amount;       
        vector<CTensor3D>     buf;
        IIterator<CTensor3D> *base_itr;
    public:
        Tensor3DBufferIterator(){ 
            base_itr = NULL; 
            do_shuffle = 0; 
            max_amount = 1 << 31;
            buf.clear();             
        }

        virtual ~Tensor3DBufferIterator(){
            if( base_itr != NULL ) {
                delete base_itr; base_itr = NULL;
            }
            for( size_t i = 0 ; i < buf.size() ; i ++ )
                tensor::free_space( buf[i] );
            buf.clear();
        }
                
        inline void set_base_itr( IIterator<CTensor3D> *base_itr ){
            this->base_itr = base_itr;
        }

        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( name, "silent") )     silent = atoi( val );
            if( !strcmp( name, "do_shuffle") ) do_shuffle = atoi( val );
            if( !strcmp( name, "max_amount") ) max_amount = atoi( val );
            base_itr->set_param( name, val );
        }
        
        virtual void init( void ){
            if( base_itr != NULL ) 
                apex_utils::error("no base iterator provided");

            int counter = max_amount;

            // buffer data into buffer 
            base_itr->before_first();
            while( base_itr->next() && counter-- > 0 ){
                CTensor3D cl;
                cl = clone( base_itr->value() );
                buf.push_back( cl );
            }
            delete base_itr; base_itr = NULL;

            if( do_shuffle ){
                if( !silent ) printf(" shuffle");
                cpu_only::shuffle( buf );
            }
            before_first();
        }
        virtual void before_first(){
            idx = 0;
        }
        virtual bool next(){
            ++ idx;
            if( idx < (int)buf.size() ) return true;
            return false;
        }
        virtual const CTensor3D &value() const{
            return buf[ idx ];
        }
    };   
};

#endif
