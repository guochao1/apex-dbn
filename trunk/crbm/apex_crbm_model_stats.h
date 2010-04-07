#ifndef _APEX_CRBM_MODEL_STATS_H_
#define _APEX_CRBM_MODEL_STATS_H_

#include "../tensor/apex_tensor.h"

namespace apex_rbm{
    // model statistics of srbm model
    struct CRBMModelStats{     
    private:
        apex_tensor::CTensor1D grad_v, grad_h;
    public:
        apex_tensor::CTensor4D grad_W;
        apex_tensor::CTensor1D pos_grad_v, pos_grad_h, neg_grad_v, neg_grad_h;
        apex_tensor::CTensor1D loss, grad_sparse;
        int sample_counter, h_size, v_size;
    public:
        CRBMModelStats( int v_max, int h_max, int y_max, int x_max ){
            grad_W.set_param( v_max, h_max, y_max, x_max );
            grad_v.set_param( v_max );
            grad_h.set_param( h_max );
            
            grad_W     = alloc_like( grad_W );
            grad_v     = alloc_like( grad_v );
            pos_grad_v = alloc_like( grad_v );           
            neg_grad_v = alloc_like( grad_v );
            grad_h     = alloc_like( grad_h );
            pos_grad_h = alloc_like( grad_h );
            neg_grad_h = alloc_like( grad_h );
            loss       = alloc_like( grad_v );          
            grad_sparse= alloc_like( grad_h );
            init();
        }

        ~CRBMModelStats(){            
            apex_tensor::tensor::free_space( grad_v );
            apex_tensor::tensor::free_space( grad_h );
            apex_tensor::tensor::free_space( grad_W );
            apex_tensor::tensor::free_space( pos_grad_v );
            apex_tensor::tensor::free_space( pos_grad_h );
            apex_tensor::tensor::free_space( neg_grad_v );
            apex_tensor::tensor::free_space( neg_grad_h );
            apex_tensor::tensor::free_space( loss );
            apex_tensor::tensor::free_space( grad_sparse );
        }
        
        inline void init(){
            sample_counter = 0;

            pos_grad_v = 0.0f;
            neg_grad_v = 0.0f;
            pos_grad_h = 0.0f;
            neg_grad_h = 0.0f;
            grad_W     = 0.0f;
            loss       = 0.0f;
            grad_sparse= 0.0f;
        }
        
        inline void save_summary( FILE *fo ) {
            float ss = ((float)sample_counter) * h_size;
            float vs = ((float)sample_counter) * v_size;

            grad_h = pos_grad_h + neg_grad_h;
            grad_v = pos_grad_v + neg_grad_v;
            
            fprintf( fo, "avg_loss=%f,", (float)apex_tensor::cpu_only::avg( loss ) / vs );
            fprintf( fo,"avg_h_grad=%f, avg_v_grad=%f, avg_w_grad=%f, avg_grad_sparse=%f,", 
                     (float)apex_tensor::cpu_only::avg( grad_h ) / ss,
                     (float)apex_tensor::cpu_only::avg( grad_v ) / ss,                     
                     (float)apex_tensor::cpu_only::avg( grad_W ) / ss,
                     (float)apex_tensor::cpu_only::avg( grad_sparse ) / ss ); 

            fprintf( fo, "pos_v_mean=%f, neg_v_mean=%f, pos_h_mean=%f, neg_h_mean=%f,", 
                     (float)apex_tensor::cpu_only::avg( pos_grad_v ) / vs,
                     (float)apex_tensor::cpu_only::avg( neg_grad_v ) / vs,                     
                     (float)apex_tensor::cpu_only::avg( pos_grad_h ) / ss,
                     (float)apex_tensor::cpu_only::avg( neg_grad_h ) / ss ); 
            
            fprintf( fo, "w_grad_max=%f, w_grad_min=%f, w_grad_std_var=%f\n",
                     (float)apex_tensor::cpu_only::max_value( grad_W ) / ss,
                     (float)apex_tensor::cpu_only::min_value( grad_W ) / ss,
                     (float)apex_tensor::cpu_only::std_var( grad_W ) / ss       );

            fflush( fo );
        }                        
        
        inline void save_detail( FILE *fo ) { 
            float ss = ((float)sample_counter) * h_size;
            for( int i = 0 ; i < pos_grad_h.x_max ; i ++ ){
                fprintf( fo,"%d:%.4f,%f,%f=%f,%f\t", i, 
                         (float) pos_grad_h[i]/ss, 
                         (float) (pos_grad_h[i]+neg_grad_h[i])/ss, 
                         (float) grad_sparse[i]  / ss ,
                         (float) apex_tensor::cpu_only::avg( grad_W[0][i] ) / ss,
                         (float) apex_tensor::cpu_only::std_var( grad_W[0][i])/ss );                
            }
            fprintf( fo, "\n" );
            fflush( fo );
        }               
    };
};
#endif
