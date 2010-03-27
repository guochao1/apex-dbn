#ifndef _APEX_TENSOR_UPDATE_TASK_H_
#define _APEX_TENSOR_UPDATE_TASK_H_

// task that use tensor input to update the inner state of model 
#include <ctime>
#include <cstring>
#include "../apex_task.h"
#include "../apex_config.h"
#include "../apex_tensor_iterator.h"
#include "../../tensor/apex_tensor.h"

namespace apex_utils{
    // model updater that updates the model 
    template<typename T>
    class ITensorUpdater{
    public:
        // set parameter necessary
        virtual void set_param( const char *name, const char *val )=0;
        // initalize the updater 
        virtual void init( void ) = 0;
        // update the model using a trunk of tensor 
        virtual void train_update_trunk( const T &data )=0;
        // validate the model using a trunk of tensor 
        virtual void validate_trunk    ( const T &data )=0;
        // end of a round
        virtual void round_end() = 0;
        // end of all training rounds
        virtual void all_end()   =0;
    };        
    
    template<typename T>
    class TensorUpdateTask : public ITask{
    private:
        ITensorUpdater<T>   *updater;
        ITensorIterator<T>  *iter;
    private:
        int task;
        int silent, do_validation;
        int trunk_size;
        int num_round, print_step;
        char name_config[ 256 ];
        
        inline void set_param_inner( const char *name, const char *val ){
            if( !strcmp( name, "silent") )        silent     = atoi( val );
            if( !strcmp( name, "do_validation") ) do_validation = atoi( val );
            if( !strcmp( name, "num_round") )  num_round  = atoi( val );
            if( !strcmp( name, "print_step") ) print_step = atoi( val );
            if( !strcmp( name, "trunk_size") ) trunk_size = atoi( val );
            iter->set_param( name, val );
            updater->set_param( name, val );
        }
    
        inline void configure(){
            apex_utils::ConfigIterator cfg( name_config );
            while( cfg.next() ){
                set_param_inner( cfg.name(), cfg.val() );
            }        
        }
        
        inline void reset_default_param(){
            silent     = 0;
            do_validation = 0;
            task       = 0;
            num_round  = 100;
            print_step = 30;
            strcpy( name_config, "config.conf" );
        }
    private:
        inline void task_update(){
            iter->init();
            updater->init();
            
            if( !silent ){
                printf("initializing end, start updating\n");
            }
            clock_t start   = clock();
            double  elapsed = 0, valid_elapsed = 0;
            // do validation for the initial model
            if( do_validation ){
                clock_t valid_start = clock();
                updater->validate_trunk( iter->validation_trunk() );
                valid_elapsed += (double)(clock() - valid_start)/CLOCKS_PER_SEC; 
            }
            
            for( int i = 0 ; i < num_round ; i ++ ){
                int sample_counter = 0;
                iter->before_first();
                while( iter->next_trunk() ){
                    updater->train_update_trunk( iter->trunk() );
                    if( ++ sample_counter  % print_step == 0 ){
                        if( !silent ){
                            printf("\r                                                               \r");
                            printf("round %8d:[%8d] %.3lf sec elapsed, %.3lf sec for validation", i , sample_counter*trunk_size, elapsed, valid_elapsed );
                            fflush( stdout );
                        }
                    }
                }
                updater->round_end();
                if( do_validation ){
                    clock_t valid_start = clock();
                    updater->validate_trunk( iter->validation_trunk() );
                    valid_elapsed += (double)(clock() - valid_start)/CLOCKS_PER_SEC; 
                }
                elapsed = (double)(clock() - start)/CLOCKS_PER_SEC; 
            }
            
            updater->all_end();
            if( !silent ){
                printf("\nupdating end, %lf sec in all, %lf for validation\n", elapsed , valid_elapsed );
            }
        }
    public:
        TensorUpdateTask( ITensorUpdater<T> *updater, ITensorIterator<T> *iter ){
            this->updater = updater;
            this->iter    = iter;   
            this->reset_default_param();
        }    

        virtual void set_param( const char *name , const char *val ) {
            set_param_inner( name, val );
        }
        
        virtual void set_task ( const char *task ){
            strcpy( name_config , task );  
        }

        virtual void print_task_help( FILE *fo ) const {
            fprintf( fo , "Usage: <config file> [param1=xxx] ...\n" );
        }
        
        virtual void run_task( void ){
            configure();        
            switch( task ){
            case 0: task_update(); break;
            default: apex_utils::error("unkown task");
            }
        }	
        virtual ~TensorUpdateTask(){}
    };
};
#endif

