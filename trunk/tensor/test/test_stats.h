#ifndef _TEST_STATS_
#define _TEST_STATS_

#include "../apex_tensor.h"

namespace apex_tensor{
    // test statistics 
    // used to esimtate the accuracy 
    // and speed

    template<typename T>
    struct TestStats{
        const char *name_A,*name_B;
        double time_A, time_B;
       
        T    abs_err_rel;
        T    abs_err_relT;
        T    abs_err;
        int  sample_counter;
        
        TestStats( const char *name_A, const char *name_B ){
            this->name_A = name_A;
            this->name_B = name_B;
            this->sample_counter = 0;
        }
        
        void init(){
            tensor::alloc_space( abs_err );
            tensor::alloc_space( abs_err_rel );
            tensor::alloc_space( abs_err_relT );
            abs_err = 0.0f;
            abs_err_rel = 0.0f;
            abs_err_relT= 0.0f;
        }

        ~ TestStats(){
            tensor::free_space( abs_err_rel );
            tensor::free_space( abs_err );
            tensor::free_space( abs_err_relT );
        }

        void print(){
            printf("A=%s,B=%s,time_A=%lf,time_B=%lf\n", name_A, name_B, time_A, time_B );
            printf("\tMAE=%f,MMAXE=%f,VAR_AE=%f\n", 
                   (float)cpu_only::avg( abs_err ) /sample_counter, 
                   (float)cpu_only::max_value( abs_err )/sample_counter,
                   (float)cpu_only::var( abs_err ) /sample_counter );
            printf("\tRMAE=%f,RMMAXE=%f,RVAR_AE=%f\n", 
                   (float)cpu_only::avg( abs_err_rel ) /sample_counter, 
                   (float)cpu_only::max_value( abs_err_rel )/sample_counter,
                   (float)cpu_only::var( abs_err_rel ) /sample_counter );            
            printf("\tRTMAE=%f,RTMMAXE=%f,RTVAR_AE=%f\n", 
                   (float)cpu_only::avg( abs_err_relT ) /sample_counter, 
                   (float)cpu_only::max_value( abs_err_relT )/sample_counter,
                   (float)cpu_only::var( abs_err_relT ) /sample_counter );            
        }
        
        void add_sample( T & ref, T & tester ){
            sample_counter ++;
			tensor::sadd__abs_err    ( abs_err    , tester, ref );  
			tensor::sadd__abs_err_rel( abs_err_rel, tester, ref );  
            tensor::sadd__abs_err_relT( abs_err_relT , tester, ref );
        }       
    };
};

#endif

