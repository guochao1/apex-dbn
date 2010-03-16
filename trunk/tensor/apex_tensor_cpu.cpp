#ifndef _APEX_TENSOR_CPU_CPP_
#define _APEX_TENSOR_CPU_CPP_

#include "apex_tensor.h"
#include "../external/apex_random.h"
#include <cmath>
#include <cstring>
// defintiions for tensor functions 
// tqchen

namespace apex_tensor{

    // initialize function and deconstructor
    static int init_counter = 0;
    // intialize the tensor engine for use, seed is 
    // the seed for random number generator    
    void init_tensor_engine_cpu( int seed ){
        if( init_counter ++ == 0 ){
            apex_random::seed( seed );
        }
    }
    // this function is called when the program exits
    void destroy_tensor_engine_cpu(){
        // do nothing
    }    

    // private functions used to support tensor op 
    namespace tensor{     
        inline void check_true( bool exp, const char *s ){
            if( !exp ){
                printf("error:%s\n",s ); exit( -1 );
            }
        }

        inline size_t num_bytes( CTensor1D ts ){
            return ts.pitch;
        }
        
        inline int num_line( CTensor1D ts ){
            return 1;
        }
        
        inline size_t num_header_bytes( CTensor1D ts ){
            return sizeof(int)*1;
        }
       
        inline int num_elem( CTensor1D ts ){
            return ts.x_max;
        }
        
        inline size_t num_bytes( CTensor2D ts ){
            return ts.pitch*ts.y_max;
        }
        
        inline int num_line( CTensor2D ts ){
            return ts.y_max;
        }
        
        inline size_t num_header_bytes( CTensor2D ts ){
            return sizeof(int)*2;
        }
        
        inline int num_elem( CTensor2D ts ){
            return ts.x_max * ts.y_max;
        }
        
        inline size_t num_bytes( CTensor3D ts ){
            return ts.pitch*ts.y_max*ts.z_max;
        }
        
        inline int num_line( CTensor3D ts ){
            return ts.y_max*ts.z_max;
        }
        
        inline size_t num_header_bytes( CTensor3D ts ){
            return sizeof(int)*3;
        }
        
        inline int num_elem( CTensor3D ts ){
            return ts.x_max * ts.y_max * ts.z_max;
        }

        inline size_t num_bytes( CTensor4D ts ){
            return ts.pitch*ts.y_max*ts.z_max*ts.h_max;
        }
        
        inline int num_line( CTensor4D ts ){
            return ts.y_max*ts.z_max*ts.h_max;
        }
        
        inline size_t num_header_bytes( CTensor4D ts ){
            return sizeof(int)*4;
        }
        
        inline int num_elem( CTensor4D ts ){
            return ts.x_max * ts.y_max *ts.z_max * ts.h_max;
        }

        inline TENSOR_FLOAT *get_line( TENSOR_FLOAT *elem, size_t pitch, size_t idx ){
            return (TENSOR_FLOAT*)((char*)elem + idx*pitch);
        }
        
        template<typename T> 
        inline TENSOR_FLOAT *get_line( T &ts, size_t idx ){
            return get_line( ts.elem, ts.pitch, idx );
        }
                        
        template<typename T> 
        inline const TENSOR_FLOAT *get_line_const( const T &ts, size_t idx ){
            return (const TENSOR_FLOAT*)((const char*)ts.elem + idx*ts.pitch);
        }
    };
    
    namespace cpu_only{
#define APEX_REDUCE_ALL_CPU_ONLY(func_name,init,rd)                     \
        TENSOR_FLOAT func_name( const T & ts ){                         \
            TENSOR_FLOAT ans = init;                                    \
            for( int i = 0 ; i < tensor::num_line( ts ) ; i ++ ){       \
                const TENSOR_FLOAT *a = tensor::get_line_const( ts, i ); \
                for( int j = 0 ; j < ts.x_max ; j ++ )                  \
                    rd;                                                 \
            }                                                           \
            return ans;                                                 \
        }                                                               \
        
        template<typename T>
        APEX_REDUCE_ALL_CPU_ONLY( sum_template, 0.0f       , ans += a[j] );
        template<typename T>
        APEX_REDUCE_ALL_CPU_ONLY( min_value_template, ts.elem[0] , if( ans > a[j] ) ans = a[j];  );
        template<typename T>
        APEX_REDUCE_ALL_CPU_ONLY( max_value_template, ts.elem[0] , if( ans < a[j] ) ans = a[j];  );
        
        template<typename T>
        TENSOR_FLOAT avg_template( const T & ts ){
            return sum_template( ts ) / tensor::num_elem( ts );
        }

#define APEX_USE_TEMPLATE_A_CPU_ONLY(func_name)                         \
        TENSOR_FLOAT func_name( const CTensor1D &dst ){                 \
            return func_name##_template( dst );                         \
        }                                                               \
        TENSOR_FLOAT func_name( const CTensor2D &dst ){                 \
            return func_name##_template( dst );                         \
        }                                                               \
        TENSOR_FLOAT func_name( const CTensor3D &dst ){                 \
            return func_name##_template( dst );                         \
        }                                                               \
        TENSOR_FLOAT func_name( const CTensor4D &dst  ){                \
            return func_name##_template( dst );                         \
        }                                                               \
        
        APEX_USE_TEMPLATE_A_CPU_ONLY( avg )
        APEX_USE_TEMPLATE_A_CPU_ONLY( max_value )
        APEX_USE_TEMPLATE_A_CPU_ONLY( min_value )
        
    };
    
    // template functions
    namespace tensor{
        // allocate space
        template<typename T>
        inline void alloc_space_template( T &ts ){
            ts.pitch= ts.x_max * sizeof(TENSOR_FLOAT);
            ts.elem = new TENSOR_FLOAT[ num_bytes( ts )/sizeof(TENSOR_FLOAT) ];
        }
        template<typename T>
        inline void free_space_template( T &ts ){
            delete[] ts.elem;
        }

        //save tensor to binary file 
        template<typename T>
        inline void save_to_file_template( const T &ts, FILE *dst ){
            fwrite( &ts, num_header_bytes( ts ) , 1 , dst );
            for( int i = 0 ; i < num_line( ts ) ; i ++ ){
                const TENSOR_FLOAT *a = get_line_const( ts, i );
                fwrite( a, sizeof( TENSOR_FLOAT ) , ts.x_max , dst ); 
            }
        }
        // load tensor from file 
        template<typename T>
        inline void load_from_file_template( T &ts, FILE *src ){
            check_true( fread( &ts, num_header_bytes( ts ) , 1 , src ) > 0, "load tensor from file" );
            alloc_space( ts );
            
            for( int i = 0 ; i < num_line( ts ) ; i ++ ){
                TENSOR_FLOAT *a = get_line( ts, i );
                check_true( fread( a, sizeof( TENSOR_FLOAT ) , ts.x_max , src ) > 0, "load tensor from file" );
            }
        }

        template<typename T>
        inline void copy_template( T &dst, const T &src ){            
            for( int i = 0 ; i < num_line( src ) ; i ++ ){
                TENSOR_FLOAT *d = get_line( dst, i );
                const TENSOR_FLOAT *s = get_line_const( src, i );
                memcpy( d, s, sizeof( TENSOR_FLOAT ) * src.x_max );
            }
        }
        
#define APEX_ELEMENTWISE_ASSIGN_OP(func_name,param,op)             \
        inline void func_name( T &dst, param ){                    \
            for( int i = 0 ; i < num_line( dst ) ; i ++ ){         \
                TENSOR_FLOAT *d = get_line( dst, i );              \
                for( int j = 0 ; j < dst.x_max ; j ++ )            \
                    op;                                            \
            }                                                      \
        }                                                          \
        
#define APEX_ELEMENTWISE_UNARY_OP(func_name,param,op)               \
        inline void func_name( T &dst, const T&src, param ){        \
            for( int i = 0 ; i < num_line( dst ) ; i ++ ){          \
                TENSOR_FLOAT *d = get_line( dst, i );               \
                const TENSOR_FLOAT *a = get_line_const( src, i );   \
                for( int j = 0 ; j < dst.x_max ; j ++ )             \
                    op;                                             \
            }                                                       \
        }                                                           \

#define APEX_ELEMENTWISE_MAP_OP(func_name,op)                       \
        inline void func_name( T &dst, const T &src ){              \
            for( int i = 0 ; i < num_line( dst ) ; i ++ ){          \
                TENSOR_FLOAT *d = get_line( dst, i );               \
                const TENSOR_FLOAT *a = get_line_const( src, i );   \
                for( int j = 0 ; j < dst.x_max ; j ++ )             \
                    op;                                             \
            }                                                       \
        }                                                           \

#define APEX_ELEMENTWISE_BINARY_OP(func_name,op)                        \
        inline void func_name( T &dst, const T &srca, const T &srcb ){  \
            for( int i = 0 ; i < num_line( dst ) ; i ++ ){              \
                TENSOR_FLOAT *d = get_line( dst ,i );                   \
                const TENSOR_FLOAT *a = get_line_const( srca, i );      \
                const TENSOR_FLOAT *b = get_line_const( srcb, i );      \
                for( int j = 0 ; j < dst.x_max ; j ++ )                 \
                    op;                                                 \
            }                                                           \
		}                                                               \

#define APEX_ELEMENTWISE_BINARY_OP_WITH_PARAM(func_name,param1,param2,op ) \
        inline void func_name( T &dst, const T &srca, const T &srcb, param1,param2 ){ \
            for( int i = 0 ; i < num_line( dst ) ; i ++ ){           \
                TENSOR_FLOAT *d = get_line( dst ,i );                   \
                const TENSOR_FLOAT *a = get_line_const( srca, i );      \
                const TENSOR_FLOAT *b = get_line_const( srcb, i );      \
                for( int j = 0 ; j < dst.x_max ; j ++ )              \
                    op;                                                 \
            }                                                           \
		}                                                               \

        template<typename T>
        APEX_ELEMENTWISE_ASSIGN_OP ( fill_template, TENSOR_FLOAT val ,d[j] = val  );
        template<typename T>
        APEX_ELEMENTWISE_ASSIGN_OP ( sample_gaussian_template, TENSOR_FLOAT sd ,d[j] = (TENSOR_FLOAT)apex_random::sample_normal()*sd  );
        template<typename T>
        APEX_ELEMENTWISE_UNARY_OP ( add_template  , TENSOR_FLOAT val ,d[j] = a[j] + val  );
        template<typename T>
        APEX_ELEMENTWISE_UNARY_OP ( mul_template  , TENSOR_FLOAT val ,d[j] = a[j] * val  );
        template<typename T>
        APEX_ELEMENTWISE_UNARY_OP ( sample_gaussian_template , TENSOR_FLOAT sd ,d[j] = (TENSOR_FLOAT)apex_random::sample_normal( a[j], sd ));
        template<typename T>
        APEX_ELEMENTWISE_MAP_OP   ( sigmoid_template      , d[j] = (TENSOR_FLOAT)(1.0/(1+exp(-a[j]))) );
        template<typename T>
        APEX_ELEMENTWISE_MAP_OP   ( sample_binary_template, d[j] = (TENSOR_FLOAT)apex_random::sample_binary( a[j] ) );
        template<typename T>
        APEX_ELEMENTWISE_BINARY_OP( add_template, d[j] = a[j]+b[j]);
        template<typename T>
        APEX_ELEMENTWISE_BINARY_OP( mul_template, d[j] = a[j]*b[j]);
        template<typename T>
        APEX_ELEMENTWISE_BINARY_OP( sub_template, d[j] = a[j]-b[j]);
        template<typename T>
        APEX_ELEMENTWISE_BINARY_OP_WITH_PARAM( scale_add_template, TENSOR_FLOAT sa, TENSOR_FLOAT sb, d[j] = sa*a[j]+sb*b[j] );
        
    };


    // definition of macros
    namespace tensor{

#define APEX_USE_TEMPLATE_A(func_name)                                  \
        void func_name( CTensor1D &dst ){                               \
            func_name##_template( dst );                                \
        }                                                               \
        void func_name( CTensor2D &dst ){                               \
            func_name##_template( dst );                                \
        }                                                               \
        void func_name( CTensor3D &dst ){                               \
            func_name##_template( dst );                                \
        }                                                               \
        void func_name( CTensor4D &dst  ){                              \
            func_name##_template( dst );                                \
        }                                                               \

#define APEX_USE_TEMPLATE_B(func_name,param,arg,cc)                     \
        void func_name( cc CTensor1D &a, param ){                       \
            func_name##_template( a, arg );                             \
        }                                                               \
        void func_name( cc CTensor2D &a, param ){                       \
            func_name##_template( a, arg );                             \
        }                                                               \
        void func_name( cc CTensor3D &a, param ){                       \
            func_name##_template( a, arg );                             \
        }                                                               \
        void func_name( cc CTensor4D &a, param ){                       \
            func_name##_template( a, arg );                             \
        }                                                               \


#define APEX_USE_TEMPLATE_C(func_name)                                  \
        void func_name( CTensor1D &dst , const CTensor1D &a, const CTensor1D &b ){ \
            func_name##_template( dst, a, b );                          \
        }                                                               \
        void func_name( CTensor2D &dst , const CTensor2D &a, const CTensor2D &b ){ \
            func_name##_template( dst, a, b );                          \
        }                                                               \
        void func_name( CTensor3D &dst , const CTensor3D &a, const CTensor3D &b ){ \
            func_name##_template( dst, a, b );                          \
        }                                                               \
        void func_name( CTensor4D &dst , const CTensor4D &a, const CTensor4D &b ){ \
            func_name##_template( dst, a, b );                          \
        }                                                               \

#define APEX_USE_TEMPLATE_D(func_name,param,arg)                        \
        void func_name( CTensor1D &dst , const CTensor1D &a, param ){   \
            func_name##_template( dst, a, arg );                        \
        }                                                               \
        void func_name( CTensor2D &dst , const CTensor2D &a, param ){   \
            func_name##_template( dst, a, arg );                        \
        }                                                               \
        void func_name( CTensor3D &dst , const CTensor3D &a, param ){   \
            func_name##_template( dst, a, arg );                        \
        }                                                               \
        void func_name( CTensor4D &dst , const CTensor4D &a, param ){   \
            func_name##_template( dst, a, arg );                        \
        }                                                               \

#define APEX_USE_TEMPLATE_E(func_name)                                  \
        void func_name( CTensor1D &dst , const CTensor1D &a){           \
            func_name##_template( dst, a );                             \
        }                                                               \
        void func_name( CTensor2D &dst , const CTensor2D &a ){          \
            func_name##_template( dst, a );                             \
        }                                                               \
        void func_name( CTensor3D &dst , const CTensor3D &a ){          \
            func_name##_template( dst, a );                             \
        }                                                               \
        void func_name( CTensor4D &dst , const CTensor4D &a ){          \
            func_name##_template( dst, a );                             \
        }                                                               \

#define APEX_USE_TEMPLATE_F(func_name)                                  \
        void func_name( CTensor1D &dst , const CTensor1D &a, const CTensor1D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb ){ \
            func_name##_template( dst, a, b, sa, sb );                  \
        }                                                               \
        void func_name( CTensor2D &dst , const CTensor2D &a, const CTensor2D &b ,TENSOR_FLOAT sa, TENSOR_FLOAT sb ){ \
            func_name##_template( dst, a, b, sa, sb );                  \
        }                                                               \
        void func_name( CTensor3D &dst , const CTensor3D &a, const CTensor3D &b ,TENSOR_FLOAT sa, TENSOR_FLOAT sb ){ \
            func_name##_template( dst, a, b, sa, sb );                  \
        }                                                               \
        void func_name( CTensor4D &dst , const CTensor4D &a, const CTensor4D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb ){ \
            func_name##_template( dst, a, b, sa, sb );                  \
        }                                                               \

    };
    // interface funtions 
    namespace tensor{
        // alloc_spaceate space for given tensor
        APEX_USE_TEMPLATE_A( alloc_space )
        APEX_USE_TEMPLATE_A( free_space  )
        APEX_USE_TEMPLATE_B( fill, TENSOR_FLOAT val, val,  )
        APEX_USE_TEMPLATE_B( sample_gaussian, TENSOR_FLOAT sd , sd,  )
        APEX_USE_TEMPLATE_B( save_to_file   , FILE *dst_file, dst_file, const )
		APEX_USE_TEMPLATE_B( load_from_file , FILE *src_file, src_file, )
        APEX_USE_TEMPLATE_C( add )
        APEX_USE_TEMPLATE_C( sub )
        APEX_USE_TEMPLATE_C( mul )
        APEX_USE_TEMPLATE_D( add, TENSOR_FLOAT val, val )
        APEX_USE_TEMPLATE_D( mul, TENSOR_FLOAT val, val )
        APEX_USE_TEMPLATE_D( sample_gaussian, TENSOR_FLOAT sd, sd )
        APEX_USE_TEMPLATE_E( sigmoid )
        APEX_USE_TEMPLATE_E( sample_binary )
        APEX_USE_TEMPLATE_E( copy )
        APEX_USE_TEMPLATE_F( scale_add )        
    };

	namespace tensor{
    // definition of macros
#define APEX_SUPPORT_DOT_1D(func_name,op1,op2)                          \
        void func_name( CTensor1D &dst, const CTensor1D &srca, const CTensor2D &srcb ){ \
            for( int i = 0; i < dst.x_max; i ++){                       \
	    		TENSOR_FLOAT tmp = 0;                                   \
				for( int j = 0; j < srca.x_max; j ++)                   \
					op1;                                                \
				dst[i] op2 tmp;                                         \
	    	}                                                           \
		}                                                               \
	
#define APEX_SUPPORT_DOT_2D(func_name)                                  \
        void func_name( CTensor2D &dst , const CTensor2D &srca, const CTensor2D &srcb ){ \
            for( int i = 0; i < num_line( dst ); i ++ ){                \
                CTensor1D dd = dst[i];                                  \
                func_name( dd, srca[i], srcb );                         \
            }                                                           \
        }                                                               \

#define APEX_SUPPORT_DOT_LT_1D(func_name,op)                            \
        void func_name( CTensor2D &dst, const CTensor1D &srca, const CTensor1D &srcb ){ \
            for( int i = 0; i < num_line( dst ); i ++ ){                \
                for( int j = 0; j < dst.x_max; j ++ )                   \
                    dst[i][j] op srca[i] * srcb[j];                     \
            }                                                           \
		}                                                               \

#define APEX_SUPPORT_DOT_LT_2D(func_name,op)                               						\
        inline void func_name( CTensor2D &dst, const CTensor2D &srca, const CTensor2D &srcb ){ 	\
            for( int i = 0; i < num_line( dst ); i ++){                    						\
				for( int j = 0; j < dst.x_max; j ++) {						               		\
					TENSOR_FLOAT tmp = 0;														\
					for( int k = 0; k < num_line( srca ); ++k)									\
						tmp += srca[k][i] * srcb[k][j]; 										\
					dst[i][j] op tmp;															\
				}																				\
			}                                                               					\
		}																						\

    };

	namespace tensor{
        //support dot operation
		APEX_SUPPORT_DOT_1D( dot    , tmp += srca[j]*srcb[j][i] , =  )
		APEX_SUPPORT_DOT_1D( add_dot, tmp += srca[j]*srcb[j][i] , += )
		APEX_SUPPORT_DOT_1D( sub_dot, tmp += srca[j]*srcb[j][i] , -= )
        
        APEX_SUPPORT_DOT_2D( dot )                          
        APEX_SUPPORT_DOT_2D( add_dot )
		APEX_SUPPORT_DOT_2D( sub_dot )

   		APEX_SUPPORT_DOT_1D( dot_rt    , tmp += srca[j]*srcb[i][j] , =  )
		APEX_SUPPORT_DOT_1D( add_dot_rt, tmp += srca[j]*srcb[i][j] , += )
		APEX_SUPPORT_DOT_1D( sub_dot_rt, tmp += srca[j]*srcb[i][j] , -= )

        APEX_SUPPORT_DOT_2D( dot_rt )                          
        APEX_SUPPORT_DOT_2D( add_dot_rt )
		APEX_SUPPORT_DOT_2D( sub_dot_rt )

		APEX_SUPPORT_DOT_LT_1D( dot_lt    , =  )
		APEX_SUPPORT_DOT_LT_1D( add_dot_lt, += )
		APEX_SUPPORT_DOT_LT_1D( sub_dot_lt, -= )

		APEX_SUPPORT_DOT_LT_2D( dot_lt    , =  )
		APEX_SUPPORT_DOT_LT_2D( add_dot_lt, += )
		APEX_SUPPORT_DOT_LT_2D( sub_dot_lt, -= )
 
    };
};
#endif
