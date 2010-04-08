#ifndef _APEX_SRBM_MODEL_STATS_H_
#define _APEX_SRBM_MODEL_STATS_H_

#include "../tensor/apex_tensor.h"

namespace apex_rbm{
    // model statistics of srbm model
    struct SRBMModelStats{     
    private:
        apex_tensor::CTensor1D grad_v, grad_h;
		apex_tensor::CTensor1D grad_l;			// gavinhu: label gradient
    public:
        apex_tensor::CTensor2D grad_W;
        apex_tensor::CTensor1D pos_grad_v, pos_grad_h, neg_grad_v, neg_grad_h;
        apex_tensor::CTensor1D loss;

		// gavinhu
		apex_tensor::CTensor2D grad_Wlh;
		apex_tensor::CTensor1D pos_grad_l, neg_grad_l;
		apex_tensor::CTensor1D lloss;

		int sample_counter;
    public:
        SRBMModelStats( int v_max, int h_max, int l_max = 0 ){
            grad_W.set_param( v_max, h_max );
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

			// gavinhu
			grad_l.set_param(l_max);
			grad_Wlh.set_param(l_max, h_max);
			if (l_max > 0) {
				grad_Wlh   = alloc_like(grad_Wlh);
				grad_l     = alloc_like(grad_l);
				pos_grad_l = alloc_like(grad_l);
				neg_grad_l = alloc_like(grad_l);
				lloss      = alloc_like(grad_l);
			}

            init();
        }

        ~SRBMModelStats(){            
            apex_tensor::tensor::free_space( grad_v );
            apex_tensor::tensor::free_space( grad_h );
            apex_tensor::tensor::free_space( grad_W );
            apex_tensor::tensor::free_space( pos_grad_v );
            apex_tensor::tensor::free_space( pos_grad_h );
            apex_tensor::tensor::free_space( neg_grad_v );
            apex_tensor::tensor::free_space( neg_grad_h );
            apex_tensor::tensor::free_space( loss );

			// gavinhu
			if (grad_l.x_max > 0) {
				apex_tensor::tensor::free_space(grad_l);
				apex_tensor::tensor::free_space(grad_Wlh);
				apex_tensor::tensor::free_space(pos_grad_l);
				apex_tensor::tensor::free_space(neg_grad_l);
				apex_tensor::tensor::free_space(lloss);
			}
        }
        
        inline void init(){
            sample_counter = 0;

            pos_grad_v = 0.0f;
            neg_grad_v = 0.0f;
            pos_grad_h = 0.0f;
            neg_grad_h = 0.0f;
            grad_W     = 0.0f;
            loss       = 0.0f;

			// gavinhu
			if (grad_l.x_max > 0) {
				pos_grad_l = 0.0f;
				neg_grad_l = 0.0f;
				grad_Wlh   = 0.0f;
				lloss      = 0.0f;
			}
        }
        
        inline void save_summary( FILE *fo ) {
            grad_h = pos_grad_h + neg_grad_h;
            grad_v = pos_grad_v + neg_grad_v;

			// gavinhu
			if (grad_l.x_max > 0)
				grad_l = pos_grad_l + neg_grad_l;

            fprintf( fo, "avg_loss=%f, ", (float)apex_tensor::cpu_only::avg( loss ) / sample_counter);

			// gavinhu
			if (grad_l.x_max > 0)
	            fprintf( fo, "avg_label_loss=%f, ", (float)apex_tensor::cpu_only::avg( lloss ) / sample_counter);

            fprintf( fo,"avg_h_grad=%f, avg_v_grad=%f, avg_w_grad=%f, ", 
                     (float)apex_tensor::cpu_only::avg( grad_h ) / sample_counter,
                     (float)apex_tensor::cpu_only::avg( grad_v ) / sample_counter,
                     (float)apex_tensor::cpu_only::avg( grad_W ) / sample_counter ); 

			// gavinhu
			if (grad_l.x_max > 0)
				fprintf( fo, "avg_l_grad=%f, ", (float)apex_tensor::cpu_only::avg( grad_l ) / sample_counter);

            fprintf( fo, "pos_v_mean=%f, neg_v_mean=%f, pos_h_mean=%f, neg_h_mean=%f", 
                     (float)apex_tensor::cpu_only::avg( pos_grad_v ) / sample_counter,
                     (float)apex_tensor::cpu_only::avg( neg_grad_v ) / sample_counter,                     
                     (float)apex_tensor::cpu_only::avg( pos_grad_h ) / sample_counter,
                     (float)apex_tensor::cpu_only::avg( neg_grad_h ) / sample_counter ); 

			// gavinhu
			if (grad_l.x_max > 0)
				fprintf( fo, ", pos_l_mean=%f, neg_l_mean=%f",
						 (float)apex_tensor::cpu_only::avg(pos_grad_l) / sample_counter,
						 (float)apex_tensor::cpu_only::avg(neg_grad_l) / sample_counter);

			fprintf( fo, "\n" );

            fflush( fo );
        }                        
        
        inline void save_detail( FILE *fo ) { 
            // temporary nothing to output, can be modified 
            fflush( fo );
        }               
    };
};
#endif
