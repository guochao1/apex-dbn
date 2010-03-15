#ifndef _APEX_TENSOR_CPU_CPP_
#define _APEX_TENSOR_CPU_CPP_

#include "apex_tensor_cpu.h"
#include "../external/apex_random.h"

// defintiions for tensor functions 
// tqchen

namespace apex_tensor{

    // initialize function and deconstructor
   
    // intialize the tensor engine for use, seed is 
    // the seed for random number generator
    void init_tensor_engine_cpu( int seed ){
        apex_random::seed( seed );
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
        
        
        template<typename T> 
        inline TENSOR_FLOAT *get_line( T &ts, size_t idx ){
            return (TENSOR_FLOAT*)((char*)ts.elem + idx*ts.pitch);
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
        inline void func_name( T &dst, const T&src ){               \
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


#define APEX_ELEMENTTWISE_BINARY_OP_SUPPOTDOT(func_name, memop, op)                        \
	inline void func_name(T &dst, const T &srca, const TB &srcb){		\
	    size_t dst_size = num_bytes( dst );					\
	    TENSOR_FLOAT *tmp = dst.elem;					\
	    bool flag = false;							\
	    if( dst.elem == srca.elem || dst.elem == srcb.elem ){			\
		flag = true;							\
	   	tmp = new TENSOR_FLOAT[ dst_size ];			\
	    	memcpy(tmp, dst.elem, dst_size);					\
	    }									\
	    memop;								\
	    for( size_t i = 0; i < num_line( dst ); i ++){			\
		TENSOR_FLOAT *d = get_line( tmp, i );				\
		const TENSOR_FLOAT *a = get_line_const( srca, i );		\
		for( size_t j = 0; j < dst.x_max; j ++){				\
		    for (size_t k = 0; k < srca.x_max; k ++){			\
			const TENSOR_FLOAT *b = get_line_const( srcb, k );	\
			op; 							\
		    }								\
		}								\
	    }									\
	    if( flag  ){							\
	    	delete[] dst.elem;						\
	    	dst.elem = tmp;							\
	    }									\
	}									\


#define APEX_ELEMENTWISE_BINARY_OP_WITH_PARAM(func_name,param1,param2,op) \
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
        template<typename T, typename TB>
        APEX_ELEMENTTWISE_BINARY_OP_SUPPOTDOT( dot_template, memset(tmp, 0, dst_size), d[j]+=a[k]*b[j] );
        template<typename T, typename TB>
        APEX_ELEMENTTWISE_BINARY_OP_SUPPOTDOT( add_dot_template, , d[j]+=a[k]*b[j] );
        template<typename T, typename TB>
        APEX_ELEMENTTWISE_BINARY_OP_SUPPOTDOT( sub_dot_template, , d[j]-=a[k]*b[j] );
        template<typename T>
        APEX_ELEMENTWISE_BINARY_OP_WITH_PARAM( scale_add_template, TENSOR_FLOAT sa, TENSOR_FLOAT sb, d[j] = sa*a[j]+sb*b[j]);
        
    };


    // definition of macros
    namespace tensor{

#define APEX_USE_TEMPLATE_A(func_name)                                  \
        void func_name( CTensor1D &dst ){                                \
            func_name##_template( dst );                                \
        }                                                               \
        void func_name( CTensor2D &dst ){                                \
            func_name##_template( dst );                                \
        }                                                               \
        void func_name( CTensor3D &dst ){                                \
            func_name##_template( dst );                                \
        }                                                               \
        void func_name( CTensor4D &dst  ){                               \
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
        void func_name( CTensor1D &dst , const CTensor1D &a, param ){     \
            func_name##_template( dst, a, arg );                        \
        }                                                               \
        void func_name( CTensor2D &dst , const CTensor2D &a, param ){     \
            func_name##_template( dst, a, arg );                        \
        }                                                               \
        void func_name( CTensor3D &dst , const CTensor3D &a, param ){     \
            func_name##_template( dst, a, arg );                        \
        }                                                               \
        void func_name( CTensor4D &dst , const CTensor4D &a, param ){     \
            func_name##_template( dst, a, arg );                        \
        }                                                               \

#define APEX_USE_TEMPLATE_E(func_name)                                  \
        void func_name( CTensor1D &dst , const CTensor1D &a){             \
            func_name##_template( dst, a );                             \
        }                                                               \
        void func_name( CTensor2D &dst , const CTensor2D &a ){            \
            func_name##_template( dst, a );                             \
        }                                                               \
        void func_name( CTensor3D &dst , const CTensor3D &a ){            \
            func_name##_template( dst, a );                             \
        }                                                               \
        void func_name( CTensor4D &dst , const CTensor4D &a ){            \
            func_name##_template( dst, a );                             \
        }                                                               \
       void func_name( CTensor1D &dst , const CTensor2D &a ){            \
            func_name##_template( dst, a );                             \
        }                                                               \
       void func_name( CTensor2D &dst , const CTensor1D &a ){            \
            func_name##_template( dst, a );                             \
	}								\

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

#define APEX_USE_TEMPLATE_G(func_name)                                  \
        void func_name( CTensor1D &dst , const CTensor1D &a, const CTenser1D &b ){             \
            func_name##_template( dst, a, b );                             \
        }                                                               \
        void func_name( CTensor2D &dst , const CTensor2D &a, const CTenser2D &b ){            \
            func_name##_template( dst, a, b );                             \
        }                                                               \
        void func_name( CTensor1D &dst , const CTensor1D &a, const CTenser2D &b ){            \
            func_name##_template( dst, a, b );                             \
        }                                                               \
        void func_name( CTensor2D &dst , const CTensor2D &a, const CTensor1D &b ){            \
            func_name##_template( dst, a, b );                             \
	}								\

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
	APEX_USE_TEMPLATE_G( dot )
	APEX_USE_TEMPLATE_G( add_dot )
	APEX_USE_TEMPLATE_G( sub_dot )
    };
};
#endif
