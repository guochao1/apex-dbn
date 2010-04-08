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

    namespace async{
        void set_dependecy( CTensor1D &ts, int stream_id ){}
        void set_dependecy( CTensor2D &ts, int stream_id ){}
        void set_dependecy( CTensor3D &ts, int stream_id ){}
        void set_dependecy( CTensor4D &ts, int stream_id ){}
    };

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
#define APEX_REDUCE_ALL_CPU_ONLY(func_name,init,rd,last)                \
        TENSOR_FLOAT func_name( const T & ts ){                         \
            init;                                                       \
            for( int i = 0 ; i < tensor::num_line( ts ) ; i ++ ){       \
                const TENSOR_FLOAT *a = tensor::get_line_const( ts, i ); \
                for( int j = 0 ; j < ts.x_max ; j ++ ){                 \
                    rd;                                                 \
                }                                                       \
            }                                                           \
            last;                                                       \
        }                                                               \
        
#define APEX_REDUCE_ALL_CPU_ONLY_A(func_name,init,rd) APEX_REDUCE_ALL_CPU_ONLY(func_name,TENSOR_FLOAT ans=init,rd,return ans )
        
        
        template<typename T>
        APEX_REDUCE_ALL_CPU_ONLY_A( sum_template, 0.0f, ans += a[j] );
        template<typename T>
        APEX_REDUCE_ALL_CPU_ONLY_A( min_value_template, ts.elem[0] , if( ans > a[j] ) ans = a[j];  );
        template<typename T>
        APEX_REDUCE_ALL_CPU_ONLY_A( max_value_template, ts.elem[0] , if( ans < a[j] ) ans = a[j];  );
        template<typename T>
        APEX_REDUCE_ALL_CPU_ONLY( var_template, 
                                  TENSOR_FLOAT s = 0.0f; TENSOR_FLOAT ss = 0.0f , 
                                  s += a[j]; ss += a[j]*a[j] , 
                                  int n = tensor::num_elem(ts); return ss/n - (s/n)*(s/n) ) 
              
        template<typename T>
        TENSOR_FLOAT avg_template( const T & ts ){
            return sum_template( ts ) / tensor::num_elem( ts );
        }
        template<typename T>
        TENSOR_FLOAT std_var_template( const T &ts ){
            return (TENSOR_FLOAT)sqrt( var_template( ts ) );
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
        APEX_USE_TEMPLATE_A_CPU_ONLY( sum )
        APEX_USE_TEMPLATE_A_CPU_ONLY( max_value )
        APEX_USE_TEMPLATE_A_CPU_ONLY( min_value )
        APEX_USE_TEMPLATE_A_CPU_ONLY( var )
        APEX_USE_TEMPLATE_A_CPU_ONLY( std_var )        
    };
    
    namespace cpu_only{
        void shuffle( CTensor3D &data ){
            for( int i = 0 ; i < data.z_max ; i ++ ){
                int j = (int)apex_random::next_uint32( data.z_max );
                for( int y = 0 ; y < data.y_max ; y ++ )
                    for( int x = 0 ; x < data.x_max ; x ++ ){
                        TENSOR_FLOAT a = data[i][y][x];
                        data[i][y][x] = data[j][y][x];
                        data[j][y][x] = a;
                    }                        
            }                
        }
        
        void rand_extract( CTensor2D &dst, const CTensor2D &src ){
            int yy, xx;

            tensor::check_true( src.y_max >= dst.y_max && src.x_max >= dst.x_max ,"extract region bigger than orignal image"); 

            if( src.y_max == dst.y_max ) 
                yy = 0;
            else 
                yy = (int)apex_random::next_uint32( src.y_max - dst.y_max );
            if( src.x_max == dst.x_max ) 
                xx = 0;
            else 
                xx = (int)apex_random::next_uint32( src.x_max - dst.x_max );
            
            for( int y = 0 ; y < dst.y_max ; y ++ ){
                memcpy( dst[y].elem, src[yy+y].elem + xx, dst.x_max * sizeof(TENSOR_FLOAT) );  
            }
        }
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
        inline void copy_template( T dst, const T src ){            
            for( int i = 0 ; i < num_line( dst ) ; i ++ ){
                TENSOR_FLOAT *d = get_line( dst, i );
                const TENSOR_FLOAT *s = get_line_const( src, i );
                memcpy( d, s, sizeof( TENSOR_FLOAT ) * dst.x_max );
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
        APEX_ELEMENTWISE_UNARY_OP ( sadd__mul_template  , TENSOR_FLOAT val ,d[j] += a[j] * val  );
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
        
        // support for error estimation
        template<typename T>
        APEX_ELEMENTWISE_BINARY_OP( sadd__abs_err_template, d[j] += (TENSOR_FLOAT)fabs(a[j]-b[j]) );
        template<typename T>
        APEX_ELEMENTWISE_BINARY_OP( sadd__abs_err_rel_template, d[j] += (TENSOR_FLOAT)fabs( 1 - b[j]/a[j] ) );
        template<typename T>
		APEX_ELEMENTWISE_BINARY_OP( sadd__abs_err_relT_template, d[j] += (TENSOR_FLOAT)( fabs(a[j]) > 1e-5 ? fabs( 1 - b[j]/a[j]): (fabs(a[j]-b[j])/1e-5)));
            
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
        APEX_USE_TEMPLATE_D( sadd__mul, TENSOR_FLOAT val, val )
        APEX_USE_TEMPLATE_D( sample_gaussian, TENSOR_FLOAT sd, sd )
        APEX_USE_TEMPLATE_E( sigmoid )
        APEX_USE_TEMPLATE_E( sample_binary )
        APEX_USE_TEMPLATE_E( copy )
        APEX_USE_TEMPLATE_F( scale_add )        

        APEX_USE_TEMPLATE_C( sadd__abs_err )
        APEX_USE_TEMPLATE_C( sadd__abs_err_rel )
        APEX_USE_TEMPLATE_C( sadd__abs_err_relT )
    };

	namespace tensor{
    // definition of macros
#define APEX_SUPPORT_DOT_1D(func_name,op1,op2)                          \
        void func_name( CTensor1D &dst, const CTensor1D &srca, const CTensor2D &srcb ){ \
            for( int i = 0; i < dst.x_max; i ++ ){                      \
	    		TENSOR_FLOAT tmp = 0;                                   \
				for( int j = 0; j < srca.x_max; j ++ )                  \
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
		APEX_SUPPORT_DOT_1D( dot      , tmp += srca[j]*srcb[j][i] , =  )
		APEX_SUPPORT_DOT_1D( sadd__dot, tmp += srca[j]*srcb[j][i] , += )
		APEX_SUPPORT_DOT_1D( ssub__dot, tmp += srca[j]*srcb[j][i] , -= )
        
        APEX_SUPPORT_DOT_2D( dot )                          
        APEX_SUPPORT_DOT_2D( sadd__dot )
		APEX_SUPPORT_DOT_2D( ssub__dot )

   		APEX_SUPPORT_DOT_1D( dot_rt      , tmp += srca[j]*srcb[i][j] , =  )
		APEX_SUPPORT_DOT_1D( sadd__dot_rt, tmp += srca[j]*srcb[i][j] , += )
		APEX_SUPPORT_DOT_1D( ssub__dot_rt, tmp += srca[j]*srcb[i][j] , -= )

        APEX_SUPPORT_DOT_2D( dot_rt )                          
        APEX_SUPPORT_DOT_2D( sadd__dot_rt )
		APEX_SUPPORT_DOT_2D( ssub__dot_rt )

		APEX_SUPPORT_DOT_LT_1D( dot_lt      , =  )
		APEX_SUPPORT_DOT_LT_1D( sadd__dot_lt, += )
		APEX_SUPPORT_DOT_LT_1D( ssub__dot_lt, -= )

		APEX_SUPPORT_DOT_LT_2D( dot_lt      , =  )
		APEX_SUPPORT_DOT_LT_2D( sadd__dot_lt, += )
		APEX_SUPPORT_DOT_LT_2D( ssub__dot_lt, -= )
 
    };

    // support for CRBM
    namespace tensor{
        namespace crbm{
            namespace store_method{
                const int SAVE = 0;
                const int ADD  = 1;
                const int SUB  = 2;
                template<int st_method>
                inline void __store( TENSOR_FLOAT &dst, TENSOR_FLOAT src );
                template<>
                inline void __store<SAVE>( TENSOR_FLOAT &dst, TENSOR_FLOAT src ){
                    dst = src;  
                }
                template<>
                inline void __store<ADD>( TENSOR_FLOAT &dst, TENSOR_FLOAT src ){
                    dst += src;  
                }
                template<>
                inline void __store<SUB>( TENSOR_FLOAT &dst, TENSOR_FLOAT src ){
                    dst -= src;  
                }
            };
            
            // fit the last two dimension of src into dst's size, copy the fitted part into dst
            void copy_fit( CTensor2D &dst, const CTensor2D &src ){
                copy_template( dst, src );
            }
            void copy_fit( CTensor3D &dst, const CTensor3D &src ){
                for( int i = 0 ; i < dst.z_max ; i ++ )
                    copy_template( dst[i], src[i] );                         
            }
            
            // normalize by maxpooling 2D
            inline void norm_maxpooling_2D_inner( CTensor2D mean, const CTensor2D energy, int pool_size ){
                for( int y = 0 ; y < mean.y_max ; y += pool_size )
                    for( int x = 0 ; x < mean.x_max ; x += pool_size ){
                        // get max value
                        TENSOR_FLOAT mmax = energy[y][x];
                        for( int dy = 0 ; dy < pool_size && y+dy < mean.y_max ; dy ++ )
                            for( int dx = 0 ; dx < pool_size && x+dx < mean.x_max ; dx ++ ){
                                if( mmax < energy[y+dy][x+dx] ) mmax = energy[y+dy][x+dx];  
                            }
                        // cal sum
                        TENSOR_FLOAT sum = 0.0f;
                        for( int dy = 0 ; dy < pool_size && y+dy < mean.y_max ; dy ++ )
                            for( int dx = 0 ; dx < pool_size && x+dx < mean.x_max ; dx ++ ){
                                mean[y+dy][x+dx] = (TENSOR_FLOAT)exp( energy[y+dy][x+dx] - mmax );
                                sum += mean[y+dy][x+dx];
                            }
                        sum += (TENSOR_FLOAT)exp( -mmax );
                        // normalize 
                        for( int dy = 0 ; dy < pool_size && y+dy < mean.y_max ; dy ++ )
                            for( int dx = 0 ; dx < pool_size && x+dx < mean.x_max ; dx ++ )
                                mean[y+dy][x+dx] /= sum;                                                    
                    } 
            } 

            void norm_maxpooling_2D( CTensor3D &mean, const CTensor3D &energy, int pool_size ){
                for( int z = 0 ; z < mean.z_max ; z ++ )
                    norm_maxpooling_2D_inner( mean[z] , energy[z], pool_size );
            }

            inline void sample_maxpooling_2D_inner( CTensor2D state, const CTensor2D mean, int pool_size ){                
                for( int y = 0 ; y < state.y_max ; y += pool_size )
                    for( int x = 0 ; x < state.x_max ; x += pool_size ){
                        bool hit = false;
                        TENSOR_FLOAT sum = 0.0f;
                        TENSOR_FLOAT rnd = (TENSOR_FLOAT)apex_random::next_double();
                        for( int dy = 0 ; dy < pool_size && y+dy < state.y_max ; dy ++ )
                            for( int dx = 0 ; dx < pool_size && x+dx < state.x_max ; dx ++ ){
                                sum += mean[y+dy][x+dx];
                                if( !hit && sum >= rnd ){
                                    state[y+dy][x+dx] = 1.0f; hit = true; 
                                }else{
                                    state[y+dy][x+dx] = 0.0f; 
                                }                                
                            }
                    }
            }

            // sample the data using 2D maxpooling 
            void sample_maxpooling_2D( CTensor3D &state, const CTensor3D &mean, int pool_size ){
                for( int z = 0 ; z < state.z_max ; z ++ )
                    sample_maxpooling_2D_inner( state[z], mean[z], pool_size );
            }

            template<int st_method>
            inline void pool_up_inner( CTensor2D dst, const CTensor2D src, int pool_size ){                
                for( int yy = 0 ; yy < dst.y_max ; yy ++ )
                    for( int xx = 0 ; xx < dst.x_max ; xx ++ ){
                        int y = yy * pool_size;
                        int x = xx * pool_size;
                        TENSOR_FLOAT sum = 0.0f;

                        for( int dy = 0 ; dy < pool_size && y+dy < src.y_max ; dy ++ )
                            for( int dx = 0 ; dx < pool_size && x+dx < src.x_max ; dx ++ ){
                                sum += src[y+dy][x+dx];
                            }
                        store_method::__store<st_method> ( dst[yy][xx] , sum );
                    }
            }

            // pool up
            void pool_up( CTensor3D &dst , const CTensor3D &src, int pool_size ){
                for( int z = 0 ; z < dst.z_max ; z ++ )
                    pool_up_inner<store_method::SAVE>( dst[z], src[z], pool_size );
            } 
            
            // 2D convolution with bias
            // convolution, leaves the valid area
            // dst = (~a) (*)  filter + bias 
            template<int st_method>
            inline void conv2_r_valid_inner( CTensor2D dst, const CTensor2D a, const CTensor2D filter ){
                for( int y = 0 ; y < dst.y_max ; y ++ )
                    for( int x = 0 ; x < dst.x_max ; x ++ ){
                        TENSOR_FLOAT sum = 0.0f;                      
                        for( int dy = 0 ; dy < filter.y_max ; dy ++ )
                            for( int dx = 0 ; dx < filter.x_max ; dx ++ ){
                                sum += a[y+dy][x+dx] * filter[dy][dx];
                            }                
                        store_method::__store<st_method>( dst[y][x] , sum );
                    }
            }
            template<int st_method>
            void conv2_r_valid_inner( CTensor3D &dst, const CTensor3D &a, const CTensor4D &filter ){
                if( st_method == store_method::SAVE ){                    
                    for( int h = 0 ; h < dst.z_max ; h ++ )
                        conv2_r_valid_inner<store_method::SAVE>( dst[h], a[0], filter[0][h] ); 

                    for( int v = 1 ; v < a.z_max ; v ++ )
                        for( int h = 0 ; h < dst.z_max ; h ++ )
                            conv2_r_valid_inner<store_method::ADD>( dst[h], a[v], filter[v][h] ); 
                }
                else{
                    for( int v = 0 ; v < a.z_max ; v ++ )
                        for( int h = 0 ; h < dst.z_max ; h ++ ){
                            conv2_r_valid_inner<st_method>( dst[h], a[v], filter[v][h] ); 
                        }

                }
            }
                        
            void conv2_r_valid     ( CTensor3D &dst, const CTensor3D &a, const CTensor4D &filter, const CTensor1D &bias ){
                for( int h = 0 ; h < dst.z_max ; h ++ )
                    dst[ h ] = bias[ h ];
                conv2_r_valid_inner<store_method::ADD>( dst, a, filter );
            }
            
            template<int st_method>
            inline void conv2_full_inner( CTensor2D dst, const CTensor2D a, const CTensor2D filter ){
                if( st_method == store_method::SAVE ) dst = 0.0f; 

                for( int y = 0 ; y < a.y_max ; y ++ )
                    for( int x = 0 ; x < a.x_max ; x ++ )
                        for( int dy = 0 ; dy < filter.y_max ; dy ++ )
                            for( int dx = 0 ; dx < filter.x_max ; dx ++ )
                                store_method::__store<st_method>( dst[y+dy][x+dx], a[y][x]*filter[dy][dx] );
            }
            
            template<int st_method>
            void conv2_full_inner( CTensor3D &dst, const CTensor3D &a, const CTensor4D &filter ){
                if( st_method == store_method::SAVE ){      
                    dst = 0.0f;
                    for( int v = 0 ; v < dst.z_max ; v ++ ){
                        for( int h = 0 ; h < a.z_max ; h ++ )
                            conv2_full_inner<store_method::ADD>( dst[v], a[h], filter[v][h] ); 
                    }
                }
                else{
                    for( int v = 0 ; v < dst.z_max ; v ++ ){
                        for( int h = 0 ; h < a.z_max ; h ++ )
                            conv2_full_inner<st_method>( dst[v], a[h], filter[v][h] ); 
                    }
                }
            }
            
            // dst = ( a) (*) filter + bias
            void conv2_full        ( CTensor3D &dst, const CTensor3D &a, const CTensor4D &filter, const CTensor1D &bias ){
                for( int v = 0 ; v < dst.z_max ; v ++ )
                    dst[v] = bias[v];
                conv2_full_inner<store_method::ADD>( dst , a, filter ); 
            }
            
            // convolution with big filter
            void sadd__conv2_r_big_filter( CTensor4D &dst, const CTensor3D &a, const CTensor3D &b ){
                for( int v = 0 ; v < a.z_max ; v ++ )
                    for( int h = 0 ; h < b.z_max ; h ++ )
                        conv2_r_valid_inner<store_method::ADD>( dst[v][h], a[v], b[h] );
            }
            void ssub__conv2_r_big_filter( CTensor4D &dst, const CTensor3D &a, const CTensor3D &b ){
                for( int v = 0 ; v < a.z_max ; v ++ )
                    for( int h = 0 ; h < b.z_max ; h ++ )
                        conv2_r_valid_inner<store_method::SUB>( dst[v][h], a[v], b[h] );
            }
            
            // sum over last two dimension
            void sadd__sum_2D( CTensor1D &dst, const CTensor3D &src ){
                for( int i = 0 ; i < dst.x_max ; i ++ )
                    dst[i] += cpu_only::sum_template( src[i] );
            }
            void ssub__sum_2D( CTensor1D &dst, const CTensor3D &src ){
                for( int i = 0 ; i < dst.x_max ; i ++ )
                    dst[i] -= cpu_only::sum_template( src[i] );
            }            

            void sum_2D    ( CTensor2D &dst, const CTensor4D &src ){
                for( int i = 0 ; i < dst.y_max ; i ++ )
                    for( int j = 0 ; j < dst.x_max ; j ++ )
                        dst[i][j] = cpu_only::sum_template( src[i][j] );
            }            
                        
            void sadd__scale( CTensor4D &dst, const CTensor2D &src, TENSOR_FLOAT scale_src ){
                for( int i = 0 ; i < src.y_max ; i ++ )
                    for( int j = 0 ; j < src.x_max ; j ++ ){
                        dst[i][j] += src[i][j] * scale_src;
                    }
            }
            
            void refill_edge_area( CTensor3D &dst, const CTensor3D &src, int edge_y_len, int edge_x_len ){
                for( int i = 0 ; i < dst.z_max ; i ++ )
                    for( int y = 0 ; y < dst.y_max ; y ++ )
                        for( int x = 0 ; x < dst.x_max ; x ++ )
                            if( y < edge_y_len || y >= dst.y_max - edge_y_len ||
                                x < edge_x_len || x >= dst.x_max - edge_x_len ){
                                    dst[i][y][x] = src[i][y][x];
                                }                            
            }
            
            // calculate information of sparse regularization
            void add_sparse_info( CTensor1D &sum_mf, CTensor1D &sum_mf_grad, const CTensor3D &src, int pool_size ){
                for( int i = 0 ; i < sum_mf.x_max ; i ++ ){
                    TENSOR_FLOAT s_mf = 0.0f, s_mf_grad = 0.0f;
                    for( int y = 0 ; y < src.y_max ; y += pool_size )
                        for( int x = 0 ; x < src.x_max ; x += pool_size ){

                            TENSOR_FLOAT sum = 0.0f;                            
                            for( int dy = 0 ; dy < pool_size && y+dy < src.y_max ; dy ++ )
                                for( int dx = 0 ; dx < pool_size && x+dx < src.x_max ; dx ++ ){
                                    sum += src[i][y+dy][x+dx];
                                }
                            s_mf += sum;
                            s_mf_grad += sum * ( 1 - sum );
                        }
                    sum_mf[i] += s_mf;
                    sum_mf_grad[i] += s_mf_grad;
                }
            }
        };        
    };
};
#endif
