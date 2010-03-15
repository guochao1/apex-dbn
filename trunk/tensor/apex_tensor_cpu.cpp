#ifndef _APEX_TENSOR_CPU_CPP_
#define _APEX_TENSOR_CPU_CPP_

#include "apex_tensor.h"
#include "../external/apex_random.h"
#include <cmath>
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
        
        inline size_t num_line( CTensor1D ts ){
            return 1;
        }
        
        inline size_t num_header_bytes( CTensor1D ts ){
            return sizeof(size_t)*1;
        }
        
        inline size_t num_bytes( CTensor2D ts ){
            return ts.pitch*ts.y_max;
        }
        
        inline size_t num_line( CTensor2D ts ){
            return ts.y_max;
        }
        
        inline size_t num_header_bytes( CTensor2D ts ){
            return sizeof(size_t)*2;
        }
        
        inline size_t num_bytes( CTensor3D ts ){
            return ts.pitch*ts.y_max*ts.z_max;
        }
        
        inline size_t num_line( CTensor3D ts ){
            return ts.y_max*ts.z_max;
        }
        
        inline size_t num_header_bytes( CTensor3D ts ){
            return sizeof(size_t)*3;
        }
        
        inline size_t num_bytes( CTensor4D ts ){
            return ts.pitch*ts.y_max*ts.z_max*ts.h_max;
        }
        
        inline size_t num_line( CTensor4D ts ){
            return ts.y_max*ts.z_max*ts.h_max;
        }
        
        inline size_t num_header_bytes( CTensor4D ts ){
            return sizeof(size_t)*4;
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
    
    // template functions
    namespace tensor{
        // allocate space
        template<typename T>
        inline void alloc_space_template( T &ts ){
            ts.pitch= ts.x_max;
            ts.elem = new TENSOR_FLOAT[ num_bytes( ts ) ];
        }
        template<typename T>
        inline void free_space_template( T &ts ){
            delete[] ts.elem;
        }

        //save tensor to binary file 
        template<typename T>
        inline void save_to_file_template( const T &ts, FILE *dst ){
            fwrite( &ts, num_header_bytes( ts ) , 1 , dst );
            for( size_t i = 0 ; i < num_line( ts ) ; i ++ ){
                const TENSOR_FLOAT *a = get_line_const( ts, i );
                fwrite( a, sizeof( TENSOR_FLOAT ) , ts.x_max , dst ); 
            }
        }
        // load tensor from file 
        template<typename T>
        inline void load_from_file_template( T &ts, FILE *src ){
            check_true( fread( &ts, num_header_bytes( ts ) , 1 , src ) > 0, "load tensor from file" );
            alloc_space( ts );
            
            for( size_t i = 0 ; i < num_line( ts ) ; i ++ ){
                TENSOR_FLOAT *a = get_line( ts, i );
                check_true( fread( a, sizeof( TENSOR_FLOAT ) , ts.x_max , src ) > 0, "load tensor from file" );
            }
        }
        
#define APEX_ELEMENTWISE_ASSIGN_OP(func_name,param,op)             \
        inline void func_name( T &dst, param ){                    \
            for( size_t i = 0 ; i < num_line( dst ) ; i ++ ){      \
                TENSOR_FLOAT *d = get_line( dst, i );              \
                for( size_t j = 0 ; j < dst.x_max ; j ++ )         \
                    op;                                            \
            }                                                      \
        }                                                          \
        
#define APEX_ELEMENTWISE_UNARY_OP(func_name,param,op)               \
        inline void func_name( T &dst, const T&src, param ){        \
            for( size_t i = 0 ; i < num_line( dst ) ; i ++ ){       \
                TENSOR_FLOAT *d = get_line( dst, i );               \
                const TENSOR_FLOAT *a = get_line_const( src, i );   \
                for( size_t j = 0 ; j < dst.x_max ; j ++ )          \
                    op;                                             \
            }                                                       \
        }                                                           \

#define APEX_ELEMENTWISE_MAP_OP(func_name,op)                       \
        inline void func_name( T &dst, const T &src ){              \
            for( size_t i = 0 ; i < num_line( dst ) ; i ++ ){       \
                TENSOR_FLOAT *d = get_line( dst, i );               \
                const TENSOR_FLOAT *a = get_line_const( src, i );   \
                for( size_t j = 0 ; j < dst.x_max ; j ++ )          \
                    op;                                             \
            }                                                       \
        }                                                           \


#define APEX_ELEMENTWISE_BINARY_OP(func_name,op)                        \
        inline void func_name( T &dst, const T &srca, const T &srcb ){  \
            for( size_t i = 0 ; i < num_line( dst ) ; i ++ ){           \
                TENSOR_FLOAT *d = get_line( dst ,i );                   \
                const TENSOR_FLOAT *a = get_line_const( srca, i );      \
                const TENSOR_FLOAT *b = get_line_const( srcb, i );      \
                for( size_t j = 0 ; j < dst.x_max ; j ++ )              \
                    op;                                                 \
            }                                                           \
		}                                                               \


#define APEX_ELEMENTWISE_BINARY_OP_WITH_PARAM(func_name,param1,param2,op ) \
        inline void func_name( T &dst, const T &srca, const T &srcb, param1,param2 ){ \
            for( size_t i = 0 ; i < num_line( dst ) ; i ++ ){           \
                TENSOR_FLOAT *d = get_line( dst ,i );                   \
                const TENSOR_FLOAT *a = get_line_const( srca, i );      \
                const TENSOR_FLOAT *b = get_line_const( srcb, i );      \
                for( size_t j = 0 ; j < dst.x_max ; j ++ )              \
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
        APEX_USE_TEMPLATE_D( add, TENSOR_FLOAT val, val )
        APEX_USE_TEMPLATE_D( mul, TENSOR_FLOAT val, val )
        APEX_USE_TEMPLATE_D( sample_gaussian, TENSOR_FLOAT sd, sd )
        APEX_USE_TEMPLATE_E( sigmoid )
        APEX_USE_TEMPLATE_E( sample_binary )
        APEX_USE_TEMPLATE_F( scale_add )        
    };

	namespace tensor{
    // definition of macros
#define APEX_SUPPORT_DOT_1D(func_name,op)                               \
        inline void func_name( CTensor1D &dst, const CTensor1D &srca, const CTensor2D &srcb ){ \
            for( size_t i = 0; i < dst.x_max; i ++){                    \
	    		TENSOR_FLOAT tmp = 0;                                   \
				for( size_t j = 0; j < srca.x_max; j ++)                \
					tmp += srca[j]*srcb[j][i];                          \
				dst[i] op tmp;                                          \
	    	}                                                           \
		}                                                               \
	

#define APEX_SUPPORT_DOT_2D(func_name)                                  \
        inline void func_name( CTensor2D &dst , const CTensor2D &srca, const CTensor2D &srcb ){ \
            for( size_t i = 0; i < num_line( dst ); i ++ ){             \
                CTensor1D dd = dst[i];                                  \
                func_name( dd, srca[i], srcb );                         \
            }                                                           \
        }                                                               \
	
    };

	namespace tensor{
        //support dot operation
		APEX_SUPPORT_DOT_1D( dot, = )
		APEX_SUPPORT_DOT_1D( add_dot, += )
		APEX_SUPPORT_DOT_1D( sub_dot, -= )

        APEX_SUPPORT_DOT_2D( dot )                          
        APEX_SUPPORT_DOT_2D( add_dot )
		APEX_SUPPORT_DOT_2D( sub_dot )
    };
};
#endif
