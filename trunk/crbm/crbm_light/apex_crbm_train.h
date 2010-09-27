#ifndef _APEX_CRBM_TRAIN_H_
#define _APEX_CRBM_TRAIN_H_

#include <vector>
#include <ctime>
#include <cstring>
#include <climits>

#include "apex_crbm.h"
#include "../../utils/apex_task.h"
#include "../../utils/apex_utils.h"
#include "../../utils/apex_config.h"
#include "../../tensor/apex_tensor.h"
#include "../../utils/data_set/apex_kyoto_iterator.h"
#include "../../utils/data_set/apex_mnist_iterator.h"

namespace apex_rbm{
    using namespace std;
    using namespace apex_tensor;
    using namespace apex_utils::iterator;

    // preserve
    class CRBMInferIterator: public IIterator<CTensor3D>{
    private:
        int silent;
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
            this->silent = 0;
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
            if( !strcmp( name, "silent") )     silent = atoi( val );
            base_itr->set_param( name, val );
        }
        virtual void init(){
            base_itr->init();
            if( !silent ) {
                printf("CRBMInferIterator\n");
            } 
        }
    };
    // Filter the data to limit mininum size
    class Tensor3DFilterIterator:public IIterator<CTensor3D>{
    private:        
        int silent;
        int filter_y_min, filter_x_min;
        IIterator<CTensor3D> *base_itr;
    public:
        Tensor3DFilterIterator(){ 
            base_itr = NULL; silent = 0;
        }
        Tensor3DFilterIterator( IIterator<CTensor3D> *base_itr ){
            this->base_itr = base_itr;
        }
        virtual ~Tensor3DFilterIterator(){
            if( base_itr != NULL ) {
                delete base_itr; base_itr = NULL;
            }
        }
                
        inline void set_base_itr( IIterator<CTensor3D> *base_itr ){
            this->base_itr = base_itr;
        }

        virtual void set_param( const char *name, const char *val ){
            if( !strcmp( name, "silent") )     silent = atoi( val );
            if( !strcmp( name, "filter_y_min") ) filter_y_min = atoi( val ); 
            if( !strcmp( name, "filter_x_min") ) filter_x_min = atoi( val ); 
            base_itr->set_param( name, val );
        }
        
        virtual void init( void ){            
            apex_utils::assert_true( base_itr!=NULL, "sample_itr::no base iterator provided");
            base_itr->init();
            before_first();

            if( silent == 0 ){
                printf("FilterIterator: y_min=%d,x_min=%d\n", filter_y_min, filter_x_min );
            }
        }

        virtual void before_first(){
            base_itr->before_first();
        }
        virtual bool next(){
            while( base_itr->next() ){
                if( base_itr->value().y_max >= filter_y_min &&
                    base_itr->value().x_max >= filter_x_min ){
                    return true;
                }
            }
            return false;
        }
        virtual const CTensor3D &value() const{
            return base_itr->value();
        }
    };
    // sample data out of previous data 
    class Tensor3DSampleIterator:public IIterator<CTensor3D>{
    private:        
        int silent;
        int sample_freq;
        int sample_y_max, sample_x_max;
        CTensor3D tmp_data;
        IIterator<CTensor3D> *base_itr;
        int sample_counter;
    public:
        Tensor3DSampleIterator(){ 
            base_itr = NULL; tmp_data.elem = NULL; sample_freq = 1; silent = 0;
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
            if( !strcmp( name, "silent") )     silent = atoi( val );
            if( !strcmp( name, "sample_freq" ) ) sample_freq = atoi( val ); 
            if( !strcmp( name, "sample_y_max") ) sample_y_max = atoi( val ); 
            if( !strcmp( name, "sample_x_max") ) sample_x_max = atoi( val ); 
            base_itr->set_param( name, val );
        }
        
        virtual void init( void ){            
            apex_utils::assert_true( base_itr!=NULL, "sample_itr::no base iterator provided");
            base_itr->init();

            if( base_itr->next() ){
                tmp_data.set_param( base_itr->value().z_max, sample_y_max, sample_x_max ); 
                tensor::alloc_space( tmp_data );
            }else{
                apex_utils::error("empty base iterator");
            }
            before_first();

            if( silent == 0 ){
                printf("SampleIterator: sample_freq=%d,y_max=%d,x_max=%d\n", sample_freq, sample_y_max, sample_x_max );
            }
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
    class Tensor3DBufferIterator: public IIterator<CTensor3D>{
    private:
        int idx;
        int silent, do_shuffle, max_amount, norm_zero_mean, norm_unit_var;       
        vector<CTensor3D>     buf;
        IIterator<CTensor3D> *base_itr;
    public:
        Tensor3DBufferIterator(){ 
            base_itr = NULL; 
            do_shuffle = 0; 
            norm_zero_mean = 0;
            norm_unit_var  = 0;
            max_amount = INT_MAX;
            buf.clear();             
        }
        Tensor3DBufferIterator( IIterator<CTensor3D> *base_itr ){ 
            this->base_itr = base_itr; 
            do_shuffle = 0; 
            norm_zero_mean = 0;
            norm_unit_var  = 0;
            max_amount = INT_MAX;
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
            if( !strcmp( name, "silent") )         silent = atoi( val );
            if( !strcmp( name, "do_shuffle") )     do_shuffle = atoi( val );
            if( !strcmp( name, "max_amount") )     max_amount = atoi( val );
            if( !strcmp( name, "norm_zero_mean") ) norm_zero_mean = atoi( val );
            if( !strcmp( name, "norm_unit_var") )  norm_unit_var  = atoi( val );
            if( base_itr != NULL ) base_itr->set_param( name, val );
        }
        
        virtual void init( void ){
            apex_utils::assert_true( base_itr != NULL ,"no base iterator provided");

            int counter = max_amount;
            int z_max=0,y_max=0,x_max=0;
            // buffer data into buffer 
            base_itr->init();
            while( base_itr->next() && counter-- > 0 ){
                CTensor3D cl;
                cl = clone( base_itr->value() );
                buf.push_back( cl ); 
                if( cl.z_max > z_max ) z_max = cl.z_max;
                if( cl.y_max > y_max ) y_max = cl.y_max;
                if( cl.x_max > x_max ) x_max = cl.x_max;
            }
            delete base_itr; base_itr = NULL;
			
			if( !silent ) {
				printf("BufferIterator:max_amount=%d,count=%d,z_max=%d,y_max=%d,x_max=%d", max_amount,(int)buf.size(),z_max,y_max,x_max );	
			}

            // normalzie to zero mean
            if( norm_zero_mean ){
                for( size_t i = 0; i < buf.size() ; i ++ )                    
                    buf[i] += -cpu_only::avg( buf[i] ) ;
                if( !silent ) printf(" norm_zero_mean");

                if( norm_unit_var ){
                    for( size_t i = 0; i < buf.size() ; i ++ )                    
                        buf[i] *= 1.0f / cpu_only::std_var( buf[i] );
                    if( !silent ) printf(" norm_unit_var");
                } 
            }

            if( do_shuffle ){
				cpu_only::shuffle( buf );
                if( !silent ) printf(" shuffle");

            }
            before_first();
			
			if( !silent ) printf("\n");
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


namespace apex_rbm{
    class CRBMLightTrainTask : public apex_utils::ITask{
    private:
        // model parameter
        /* parameter for new layer */
        CRBMModelParam param_new_layer;
        CRBMTrainParam param_train;
        /* model of CDBN */
		CDBNModel  model;
    private:
        ICRBMTrainer *crbm_trainer;
        ICRBMInferencer *crbm_infer;
        IIterator<CTensor3D> *base_itr;
        CRBMModelStats *crbm_stats;
    private:
        // name of configure file
        char name_config[ 256 ];
        // 0 = new layer, 1 = continue last layer's training
        int task;
        // whether to be silent 
        int silent;
        // step of cd
        int cd_step;        
        // input model name 
        char name_model_in[ 256 ];
        // start counter of model
        int start_counter;
        // folder name of output  
        char name_model_out_folder[ 256 ];
    private:
        int do_validation, print_step;
        int num_round, train_repeat, validation_amount;
    private:
        // name for summary and detail information of validation
        char name_summary_log[256], name_detail_log[256];
        // file for summary and detail information of validation
        FILE *fo_summary_log, *fo_detail_log;        
    private:
        inline void reset_default(){
            strcpy( name_config, "config.conf" );
            strcpy( name_model_in, "NULL" );
            strcpy( name_model_out_folder, "models" );            
            strcpy( name_summary_log, "summary.log.txt" );           
            strcpy( name_detail_log , "detail.log.txt"  );            
            cd_step = 1;  
            print_step = 1;
            train_repeat = 1;
            validation_amount = 0;
            num_round = 10;            
            task = silent = start_counter = 0; do_validation = 0;
        }
    public:
        CRBMLightTrainTask(){
            crbm_trainer = NULL;
            crbm_infer   = NULL;
            base_itr     = NULL;
            crbm_stats   = NULL;
            reset_default();
        }
        virtual ~CRBMLightTrainTask(){
            if( base_itr != NULL ) delete base_itr;
            if( crbm_trainer != NULL ) delete crbm_trainer;            
            if( crbm_stats != NULL ) delete crbm_stats;
            if( crbm_infer != NULL ) delete crbm_infer;
        }
    private:
        inline void set_param_inner( const char *name, const char *val ){
            if( !strcmp( name,"task"   ))            task    = atoi( val ); 
            if( !strcmp( name,"silent" ))            silent  = atoi( val );        
            if( !strcmp( name,"cd_step" ))           cd_step = atoi( val );
            if( !strcmp( name,"start_counter" ))     start_counter = atoi( val );
            if( !strcmp( name,"model_in" ))          strcpy( name_model_in, val ); 
            if( !strcmp( name,"model_out_folder" ))  strcpy( name_model_out_folder, val ); 
            if( !strcmp( name,"summary_log" ))       strcpy( name_summary_log, val ); 
            if( !strcmp( name,"detail_log" ))        strcpy( name_detail_log , val ); 
            if( !strcmp( name,"num_round"  ))        num_round    = atoi( val ); 
            if( !strcmp( name,"train_repeat"  ))     train_repeat = atoi( val );
            if( !strcmp( name,"validation_amount" )) validation_amount  = atoi( val ); 
            if( !strcmp( name, "silent") )           silent     = atoi( val );
            if( !strcmp( name, "do_validation") )    do_validation = atoi( val );
            if( !strcmp( name, "print_step") )       print_step = atoi( val );
            param_new_layer.set_param( name, val );
            param_train.set_param    ( name, val );
            if( base_itr != NULL ) base_itr->set_param( name, val );
        }
        inline void configure(){
            apex_utils::ConfigIterator cfg( name_config );
            while( cfg.next() ){
                set_param_inner( cfg.name(), cfg.val() );
            }        
        }
        
        inline void load_model(){
            FILE *fi = apex_utils::fopen_check( name_model_in, "rb" );
            // load model from file 
            model.load_from_file( fi );
            fclose( fi );
        }
        
        inline void save_model(){
            char name[256];
            sprintf(name,"%s/%02d.%04d.model" , name_model_out_folder, (int)(model.layers.size()-1) , start_counter ++ );
            FILE *fo  = apex_utils::fopen_check( name, "wb" );            
            model.save_to_file( fo );
            fclose( fo );
        }            
        
        inline void init( void ){
            this->configure();
            if( strcmp( name_model_in,"NULL") != 0 ){
                 this->load_model();
			}                        
            if( task == 0 ){
                model.add_layer( param_new_layer );
            }
            // more than one layer, we need inferencer
            if( model.layers.size() > 1 ){
                CRBMModel md = model.layers.back();
                model.layers.pop_back();
                crbm_infer = factory::create_crbm_inferencer( model, 
                                                              param_train.input_y_max, 
                                                              param_train.input_x_max );
                // reset input size to infered size
                int z_max, y_max, x_max;
                crbm_infer->get_top_bound( z_max, y_max, x_max );
                param_train.input_y_max = y_max;
                param_train.input_x_max = x_max;
                
                if( param_train.forward_bias ) crbm_infer->forward_bias( md.v_bias );

                model.layers.push_back( md );
            }
            crbm_trainer = factory::create_crbm_trainer( model.layers.back(), param_train );
			crbm_trainer->set_cd_step( cd_step );

            // configure iterator input
            this->configure_iterator();            
            base_itr->init();
            
            // saved for further usage
            fo_summary_log = apex_utils::fopen_check( name_summary_log , "w" );
            fo_detail_log  = apex_utils::fopen_check( name_detail_log  , "w" );
        }     
        inline void round_end(){
            crbm_trainer->clone_model( model.layers.back() );
            this->save_model();
        }
        inline void all_end(){
            fclose( fo_summary_log );
            fclose( fo_detail_log );
        }
    private:
        inline void run_validation (){
            if( crbm_stats == NULL ){
                CRBMModelParam &param = model.layers.back().param;            
                crbm_stats = new CRBMModelStats( param.v_max, param.h_max, param.y_max, param.x_max );
            }

            // limit number of validation 
            LimitCounterIterator<CTensor3D> ic( base_itr, validation_amount );
            crbm_trainer->validate_stats( *crbm_stats, &ic );
            crbm_stats->save_summary( fo_summary_log );
            crbm_stats->save_detail ( fo_detail_log  );
            // reinit the stats for next usage 
            crbm_stats->init();
        }
    private:
        // whether the procedure contains infer procedure
        int infer_created;
    private:
        inline void configure_iterator(){
            infer_created = 0;
            apex_utils::ConfigIterator cfg( name_config );
            while( cfg.next() ){
                if( !strcmp( cfg.name(), "iterator_chain" ) ) create_iterator( cfg.val() );
                if( base_itr != NULL ) base_itr->set_param( cfg.name(), cfg.val() );
            }        
            if( model.layers.size() > 1 ){
                apex_utils::assert_true( infer_created!=0, "multiple layer must have inference iterator" );
            }
        }

        inline void create_iterator( const char* itr_type ){
            if( !strcmp( itr_type, "data_mnist") ){
                apex_utils::assert_true( base_itr==NULL, "already specify base iterator" );
                base_itr = new apex_utils::iterator::MNISTIterator<CTensor3D>();
                return;
            }
            if( !strcmp( itr_type, "data_kyoto") ){
                apex_utils::assert_true( base_itr==NULL, "already specify base iterator" );
                base_itr = new apex_utils::iterator::KyotoIterator<CTensor3D>();
                return;
            }

            apex_utils::assert_true( base_itr!=NULL, "must specify base iterator" );
            if( !strcmp( itr_type, "proc_sample") ){
                base_itr = new Tensor3DSampleIterator( base_itr ); 
                return;
            }
            if( !strcmp( itr_type, "proc_filter") ){
                base_itr = new Tensor3DFilterIterator( base_itr ); 
                return;
            }
            if( !strcmp( itr_type, "proc_buffer") ){
                base_itr = new Tensor3DBufferIterator( base_itr ); 
                return;
            }            
            if( !strcmp( itr_type, "proc_infer") ){                
                apex_utils::assert_true( !infer_created, "at most one inference iterator" );
                infer_created = 1; 
                if( crbm_infer != NULL ) {
                    base_itr = new CRBMInferIterator( base_itr, crbm_infer );
                    crbm_infer = NULL;
                }
                return;
            }            
        }
    public:
        virtual void set_param( const char *name , const char *val ){
            this->set_param_inner( name, val );
        }
        virtual void set_task ( const char *task ){
            strcpy( name_config, task );
        }
        virtual void print_task_help( FILE *fo ) const {
            printf("Usage:<config> [xxx=xx]\n");
        }
        virtual void run_task( void ){
            this->init();
            if( !silent ){
                printf("initializing end, start updating\n");
            }
            clock_t start   = clock();
            double  elapsed = 0, valid_elapsed = 0;
            
            for( int i = 0 ; i < num_round ; i ++ ){
                int sample_counter = 0;
                // validation procedure
                if( do_validation ){
                    clock_t valid_start = clock();
                    this->run_validation();
                    valid_elapsed += (double)(clock() - valid_start)/CLOCKS_PER_SEC; 
                }               
                // save initial model 
                this->round_end();
                for( int j = 0; j < train_repeat; j ++ ){ 
                    base_itr->before_first();
                    while( base_itr->next() ){
                        // training procedure
                        crbm_trainer->train_update( base_itr->value() );
                        if( ++ sample_counter  % print_step == 0 ){
                            if( !silent ){
                                printf("\r                                                               \r");
                                printf("round %8d:[%8d] %.3lf sec elapsed, %.3lf sec for validation", i , sample_counter, elapsed, valid_elapsed );
                                fflush( stdout );
                            }
                        }
                    }
                }
                elapsed = (double)(clock() - start)/CLOCKS_PER_SEC; 
                // end of a round
            }
            
            this->round_end();
            this->all_end();
            if( !silent ){
                printf("\nupdating end, %lf sec in all, %lf for validation\n", elapsed , valid_elapsed );
            }
        }        
    };
};

#endif
