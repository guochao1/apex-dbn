#ifndef _APEX_TENSOR_CPU_INLINE_CPP_
#define _APEX_TENSOR_CPU_INLINE_CPP_

#include <cstring>
#include "../external/apex_random.h"

#if __APEX_TENSOR_USE_BLAS__
#include "../external/cblas.h"
#endif

namespace apex_tensor{
    namespace async{
        inline void set_dependecy( CTensor1D &ts, int stream_id ){}
        inline void set_dependecy( CTensor2D &ts, int stream_id ){}
        inline void set_dependecy( CTensor3D &ts, int stream_id ){}
        inline void set_dependecy( CTensor4D &ts, int stream_id ){}
    };
    
    // private functions used to support tensor op 
    namespace cpu_template{     
        inline void check_true( bool exp, const char *s ){
            if( !exp ){
                printf("error:%s\n",s ); exit( -1 );
            }
        }

        inline unsigned int num_bytes( CTensor1D ts ){
            return ts.pitch;
        }
        
        inline int num_line( CTensor1D ts ){
            return 1;
        }
        
        inline unsigned int num_header_bytes( CTensor1D ts ){
            return sizeof(int)*1;
        }
       
        inline int num_elem( CTensor1D ts ){
            return ts.x_max;
        }
        
        inline unsigned int num_bytes( CTensor2D ts ){
            return ts.pitch*ts.y_max;
        }
        
        inline int num_line( CTensor2D ts ){
            return ts.y_max;
        }
        
        inline unsigned int num_header_bytes( CTensor2D ts ){
            return sizeof(int)*2;
        }
        
        inline int num_elem( CTensor2D ts ){
            return ts.x_max * ts.y_max;
        }
        
        inline unsigned int num_bytes( CTensor3D ts ){
            return ts.pitch*ts.y_max*ts.z_max;
        }
        
        inline int num_line( CTensor3D ts ){
            return ts.y_max*ts.z_max;
        }
        
        inline unsigned int num_header_bytes( CTensor3D ts ){
            return sizeof(int)*3;
        }
        
        inline int num_elem( CTensor3D ts ){
            return ts.x_max * ts.y_max * ts.z_max;
        }

        inline unsigned int num_bytes( CTensor4D ts ){
            return ts.pitch*ts.y_max*ts.z_max*ts.h_max;
        }
        
        inline int num_line( CTensor4D ts ){
            return ts.y_max*ts.z_max*ts.h_max;
        }
        
        inline unsigned int num_header_bytes( CTensor4D ts ){
            return sizeof(int)*4;
        }
        
        inline int num_elem( CTensor4D ts ){
            return ts.x_max * ts.y_max *ts.z_max * ts.h_max;
        }

        inline TENSOR_FLOAT *get_line( TENSOR_FLOAT *elem, unsigned int pitch, unsigned int idx ){
            return (TENSOR_FLOAT*)((char*)elem + idx*pitch);
        }
        
        template<typename T> 
        inline TENSOR_FLOAT *get_line( T &ts, unsigned int idx ){
            return get_line( ts.elem, ts.pitch, idx );
        }
                        
        template<typename T> 
        inline const TENSOR_FLOAT *get_line_const( const T &ts, unsigned int idx ){
            return (const TENSOR_FLOAT*)((const char*)ts.elem + idx*ts.pitch);
        }
    };
    
    namespace cpu_only{
#define APEX_CPU_REDUCE_ALL_CPU_ONLY(func_name,init,rd,last)                \
        inline TENSOR_FLOAT func_name( const T & ts ){                  \
            init;                                                       \
            for( int i = 0 ; i < cpu_template::num_line( ts ) ; i ++ ){      \
                const TENSOR_FLOAT *a = cpu_template::get_line_const( ts, i ); \
                for( int j = 0 ; j < ts.x_max ; j ++ ){                 \
                    rd;                                                 \
                }                                                       \
            }                                                           \
            last;                                                       \
        }                                                               \
        
#define APEX_CPU_REDUCE_ALL_CPU_ONLY_A(func_name,init,rd) APEX_CPU_REDUCE_ALL_CPU_ONLY(func_name,TENSOR_FLOAT ans=init,rd,return ans )
        
        
        template<typename T>
        APEX_CPU_REDUCE_ALL_CPU_ONLY_A( sum_template, 0.0f, ans += a[j] );
        template<typename T>
        APEX_CPU_REDUCE_ALL_CPU_ONLY_A( min_value_template, ts.elem[0] , if( ans > a[j] ) ans = a[j];  );
        template<typename T>
        APEX_CPU_REDUCE_ALL_CPU_ONLY_A( max_value_template, ts.elem[0] , if( ans < a[j] ) ans = a[j];  );
        template<typename T>
        APEX_CPU_REDUCE_ALL_CPU_ONLY( var_template, 
                                  TENSOR_FLOAT s = 0.0f; TENSOR_FLOAT ss = 0.0f , 
                                  s += a[j]; ss += a[j]*a[j] , 
                                  int n = cpu_template::num_elem(ts); return ss/n - (s/n)*(s/n) ) 
              
        template<typename T>
        TENSOR_FLOAT avg_template( const T & ts ){
            return sum_template( ts ) / cpu_template::num_elem( ts );
        }
        template<typename T>
        TENSOR_FLOAT std_var_template( const T &ts ){
            return (TENSOR_FLOAT)sqrt( var_template( ts ) );
        }
        
#define APEX_CPU_USE_TEMPLATE_A_CPU_ONLY(func_name)                         \
        inline TENSOR_FLOAT func_name( const CTensor1D &dst ){          \
            return func_name##_template( dst );                         \
        }                                                               \
        inline TENSOR_FLOAT func_name( const CTensor2D &dst ){          \
            return func_name##_template( dst );                         \
        }                                                               \
        inline TENSOR_FLOAT func_name( const CTensor3D &dst ){          \
            return func_name##_template( dst );                         \
        }                                                               \
        inline TENSOR_FLOAT func_name( const CTensor4D &dst  ){         \
            return func_name##_template( dst );                         \
        }                                                               \
        
        APEX_CPU_USE_TEMPLATE_A_CPU_ONLY( avg )
        APEX_CPU_USE_TEMPLATE_A_CPU_ONLY( sum )
        APEX_CPU_USE_TEMPLATE_A_CPU_ONLY( max_value )
        APEX_CPU_USE_TEMPLATE_A_CPU_ONLY( min_value )
        APEX_CPU_USE_TEMPLATE_A_CPU_ONLY( var )
        APEX_CPU_USE_TEMPLATE_A_CPU_ONLY( std_var )        
    };
        
    namespace cpu_only{
        inline void shuffle( CTensor4D &data ){
            for( int i = 0 ; i < data.h_max ; i ++ ){
                int j = (int)apex_random::next_uint32( data.z_max );
                for( int z = 0 ; z < data.z_max ; z ++ )
                    for( int y = 0 ; y < data.y_max ; y ++ )
                        for( int x = 0 ; x < data.x_max ; x ++ ){
                            TENSOR_FLOAT a = data[i][z][y][x];
                            data[i][z][y][x] = data[j][z][y][x];
                            data[j][z][y][x] = a;
                        }                        
            }                
        }
        
        inline void shuffle( std::vector<CTensor3D> &data ){
            for( size_t i = 0 ; i < data.size() ; i ++ ){
                int j = (int)apex_random::next_uint32( (uint32_t)data.size() );
                CTensor3D a = data[i];
                data[i] = data[j]; 
                data[j] = a;
            }
        }
        
        inline void rand_extract( CTensor3D &dst, const CTensor3D &src ){
            int yy, xx;

            cpu_template::check_true( src.y_max >= dst.y_max && src.x_max >= dst.x_max ,"extract region bigger than orignal image"); 

            if( src.y_max == dst.y_max ) 
                yy = 0;
            else 
                yy = (int)apex_random::next_uint32( src.y_max - dst.y_max );
            if( src.x_max == dst.x_max ) 
                xx = 0;
            else 
                xx = (int)apex_random::next_uint32( src.x_max - dst.x_max );

            for( int z = 0 ; z < dst.z_max ; z ++ ){
                for( int y = 0 ; y < dst.y_max ; y ++ ){
                    memcpy( dst[z][y].elem, src[z][yy+y].elem + xx, dst.x_max * sizeof(TENSOR_FLOAT) );  
                }
            }
        }
    };    

    // template functions

    namespace cpu_template{
        // allocate space
        template<typename T>
        inline void alloc_space_template( T &ts ){
            ts.pitch= ts.x_max * sizeof(TENSOR_FLOAT);
                ts.elem = new TENSOR_FLOAT[ cpu_template::num_bytes( ts )/sizeof(TENSOR_FLOAT) ];
        }
        template<typename T>
        inline void free_space_template( T &ts ){
            delete[] ts.elem;
        }
        
        //save tensor to binary file 
        template<typename T>
        inline void save_to_file_template( const T &ts, FILE *dst ){
            fwrite( &ts, cpu_template::num_header_bytes( ts ) , 1 , dst );
            for( int i = 0 ; i < cpu_template::num_line( ts ) ; i ++ ){
                const TENSOR_FLOAT *a = cpu_template::get_line_const( ts, i );
                fwrite( a, sizeof( TENSOR_FLOAT ) , ts.x_max , dst ); 
            }
        }
        // load tensor from file 
        template<typename T>
            inline void load_from_file_template( T &ts, FILE *src ){
            cpu_template::check_true( fread( &ts, cpu_template::num_header_bytes( ts ) , 1 , src ) > 0, "load tensor from file" );
            tensor::alloc_space( ts );
            
            for( int i = 0 ; i < cpu_template::num_line( ts ) ; i ++ ){
                TENSOR_FLOAT *a = cpu_template::get_line( ts, i );
                cpu_template::check_true( fread( a, sizeof( TENSOR_FLOAT ) , ts.x_max , src ) > 0, "load tensor from file" );
            }
        }
        
        template<typename T>
        inline void copy_template( T dst, const T src ){            
            for( int i = 0 ; i < cpu_template::num_line( dst ) ; i ++ ){
                TENSOR_FLOAT *d = cpu_template::get_line( dst, i );
                const TENSOR_FLOAT *s = cpu_template::get_line_const( src, i );
                memcpy( d, s, sizeof( TENSOR_FLOAT ) * dst.x_max );
            }
        }
        
#define APEX_CPU_ELEMENTWISE_ASSIGN_OP(func_name,param,op)              \
        inline void func_name( T &dst, param ){                         \
            for( int i = 0 ; i < cpu_template::num_line( dst ) ; i ++ ){ \
                TENSOR_FLOAT *d = cpu_template::get_line( dst, i );     \
                for( int j = 0 ; j < dst.x_max ; j ++ )                 \
                    op;                                                 \
            }                                                           \
            }                                                           \
        
#define APEX_CPU_ELEMENTWISE_UNARY_OP(func_name,param,op)               \
        inline void func_name( T &dst, const T&src, param ){            \
            for( int i = 0 ; i < cpu_template::num_line( dst ) ; i ++ ){ \
                TENSOR_FLOAT *d = cpu_template::get_line( dst, i );     \
                const TENSOR_FLOAT *a = cpu_template::get_line_const( src, i ); \
                for( int j = 0 ; j < dst.x_max ; j ++ )                 \
                    op;                                                 \
            }                                                           \
        }                                                               \
        
#define APEX_CPU_ELEMENTWISE_MAP_OP(func_name,op)                       \
        inline void func_name( T &dst, const T &src ){                  \
            for( int i = 0 ; i < cpu_template::num_line( dst ) ; i ++ ){ \
                TENSOR_FLOAT *d = cpu_template::get_line( dst, i );     \
                const TENSOR_FLOAT *a = cpu_template::get_line_const( src, i ); \
                for( int j = 0 ; j < dst.x_max ; j ++ )                 \
                    op;                                                 \
            }                                                           \
        }                                                               \
        
#define APEX_CPU_ELEMENTWISE_BINARY_OP(func_name,op)                    \
        inline void func_name( T &dst, const T &srca, const T &srcb ){  \
            for( int i = 0 ; i < cpu_template::num_line( dst ) ; i ++ ){ \
                TENSOR_FLOAT *d = cpu_template::get_line( dst ,i );     \
                const TENSOR_FLOAT *a = cpu_template::get_line_const( srca, i ); \
                    const TENSOR_FLOAT *b = cpu_template::get_line_const( srcb, i ); \
                    for( int j = 0 ; j < dst.x_max ; j ++ )             \
                        op;                                             \
            }                                                           \
        }                                                               \
        
#define APEX_CPU_ELEMENTWISE_BINARY_OP_WITH_PARAM(func_name,param1,param2,op ) \
        inline void func_name( T &dst, const T &srca, const T &srcb, param1,param2 ){ \
            for( int i = 0 ; i < cpu_template::num_line( dst ) ; i ++ ){ \
                TENSOR_FLOAT *d = cpu_template::get_line( dst ,i );     \
                const TENSOR_FLOAT *a = cpu_template::get_line_const( srca, i ); \
                const TENSOR_FLOAT *b = cpu_template::get_line_const( srcb, i ); \
                for( int j = 0 ; j < dst.x_max ; j ++ )                 \
                    op;                                                 \
            }                                                           \
        }                                                               \
        
        template<typename T>
        APEX_CPU_ELEMENTWISE_ASSIGN_OP ( fill_template, TENSOR_FLOAT val ,d[j] = val  );
        template<typename T>
        APEX_CPU_ELEMENTWISE_ASSIGN_OP ( sample_gaussian_template, TENSOR_FLOAT sd ,d[j] = (TENSOR_FLOAT)apex_random::sample_normal()*sd  );
        template<typename T>
        APEX_CPU_ELEMENTWISE_UNARY_OP  ( add_template  , TENSOR_FLOAT val ,d[j] = a[j] + val  );
        template<typename T>
        APEX_CPU_ELEMENTWISE_UNARY_OP  ( mul_template  , TENSOR_FLOAT val ,d[j] = a[j] * val  );
        template<typename T>
        APEX_CPU_ELEMENTWISE_UNARY_OP  ( sadd__mul_template  , TENSOR_FLOAT val ,d[j] += a[j] * val  );
        template<typename T>
        APEX_CPU_ELEMENTWISE_UNARY_OP  ( sample_gaussian_template , TENSOR_FLOAT sd ,d[j] = (TENSOR_FLOAT)apex_random::sample_normal( a[j], sd ));
        template<typename T>
        APEX_CPU_ELEMENTWISE_MAP_OP    ( sigmoid_template      , d[j] = (TENSOR_FLOAT)(1.0/(1+exp(-a[j]))) );
        template<typename T>
        APEX_CPU_ELEMENTWISE_MAP_OP    ( sample_binary_template, d[j] = (TENSOR_FLOAT)apex_random::sample_binary( a[j] ) );
        template<typename T>
        APEX_CPU_ELEMENTWISE_BINARY_OP ( add_template, d[j] = a[j]+b[j]);
        template<typename T>
        APEX_CPU_ELEMENTWISE_BINARY_OP ( mul_template, d[j] = a[j]*b[j]);
        template<typename T>
        APEX_CPU_ELEMENTWISE_BINARY_OP ( sub_template, d[j] = a[j]-b[j]);
        template<typename T>
        APEX_CPU_ELEMENTWISE_BINARY_OP_WITH_PARAM( scale_add_template, TENSOR_FLOAT sa, TENSOR_FLOAT sb, d[j] = sa*a[j]+sb*b[j] );
        template<typename T>
        APEX_CPU_ELEMENTWISE_BINARY_OP_WITH_PARAM( sadd__scale_add_template, TENSOR_FLOAT sa, TENSOR_FLOAT sb, d[j] += sa*a[j]+sb*b[j] );
        
        // support for error estimation
        template<typename T>
        APEX_CPU_ELEMENTWISE_BINARY_OP( sadd__abs_err_template, d[j] += (TENSOR_FLOAT)fabs(a[j]-b[j]) );
        template<typename T>
        APEX_CPU_ELEMENTWISE_BINARY_OP( sadd__abs_err_rel_template, d[j] += (TENSOR_FLOAT)fabs( 1 - b[j]/a[j] ) );
        template<typename T>
        APEX_CPU_ELEMENTWISE_BINARY_OP( sadd__abs_err_relT_template, d[j] += (TENSOR_FLOAT)( fabs(a[j]) > 1e-5 ? fabs( 1 - b[j]/a[j]): (fabs(a[j]-b[j])/1e-5)));        
    };
    

    // definition of macros
    namespace tensor{

#define APEX_CPU_USE_TEMPLATE_A(func_name)                              \
        inline void func_name( CTensor1D &dst ){                        \
            cpu_template::func_name##_template( dst );                  \
        }                                                               \
        inline void func_name( CTensor2D &dst ){                        \
            cpu_template::func_name##_template( dst );                  \
        }                                                               \
        inline void func_name( CTensor3D &dst ){                        \
            cpu_template::func_name##_template( dst );                  \
        }                                                               \
        inline void func_name( CTensor4D &dst  ){                       \
            cpu_template::func_name##_template( dst );                  \
        }                                                               \
        
#define APEX_CPU_USE_TEMPLATE_B(func_name,param,arg,cc)                 \
        inline void func_name( cc CTensor1D &a, param ){                \
            cpu_template::func_name##_template( a, arg );               \
        }                                                               \
        inline void func_name( cc CTensor2D &a, param ){                \
            cpu_template::func_name##_template( a, arg );               \
        }                                                               \
        inline void func_name( cc CTensor3D &a, param ){                \
            cpu_template::func_name##_template( a, arg );               \
        }                                                               \
        inline void func_name( cc CTensor4D &a, param ){                \
            cpu_template::func_name##_template( a, arg );               \
        }                                                               \
        
        
#define APEX_CPU_USE_TEMPLATE_C(func_name)                              \
        inline void func_name( CTensor1D &dst , const CTensor1D &a, const CTensor1D &b ){ \
            cpu_template::func_name##_template( dst, a, b );            \
        }                                                               \
        inline void func_name( CTensor2D &dst , const CTensor2D &a, const CTensor2D &b ){ \
            cpu_template::func_name##_template( dst, a, b );            \
        }                                                               \
        inline void func_name( CTensor3D &dst , const CTensor3D &a, const CTensor3D &b ){ \
            cpu_template::func_name##_template( dst, a, b );            \
        }                                                               \
        inline void func_name( CTensor4D &dst , const CTensor4D &a, const CTensor4D &b ){ \
            cpu_template::func_name##_template( dst, a, b );            \
        }                                                               \
        
#define APEX_CPU_USE_TEMPLATE_D(func_name,param,arg)                    \
        inline void func_name( CTensor1D &dst , const CTensor1D &a, param ){ \
            cpu_template::func_name##_template( dst, a, arg );          \
        }                                                               \
        inline void func_name( CTensor2D &dst , const CTensor2D &a, param ){ \
            cpu_template::func_name##_template( dst, a, arg );          \
        }                                                               \
        inline void func_name( CTensor3D &dst , const CTensor3D &a, param ){ \
            cpu_template::func_name##_template( dst, a, arg );          \
        }                                                               \
        inline void func_name( CTensor4D &dst , const CTensor4D &a, param ){ \
            cpu_template::func_name##_template( dst, a, arg );          \
        }                                                               \
        
#define APEX_CPU_USE_TEMPLATE_E(func_name)                              \
        inline void func_name( CTensor1D &dst , const CTensor1D &a){    \
            cpu_template::func_name##_template( dst, a );               \
        }                                                               \
        inline void func_name( CTensor2D &dst , const CTensor2D &a ){   \
            cpu_template::func_name##_template( dst, a );               \
        }                                                               \
        inline void func_name( CTensor3D &dst , const CTensor3D &a ){   \
            cpu_template::func_name##_template( dst, a );               \
        }                                                               \
        inline void func_name( CTensor4D &dst , const CTensor4D &a ){   \
            cpu_template::func_name##_template( dst, a );               \
        }                                                               \
        
#define APEX_CPU_USE_TEMPLATE_F(func_name)                              \
        inline void func_name( CTensor1D &dst , const CTensor1D &a, const CTensor1D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb ){ \
            cpu_template::func_name##_template( dst, a, b, sa, sb );    \
        }                                                               \
        inline void func_name( CTensor2D &dst , const CTensor2D &a, const CTensor2D &b ,TENSOR_FLOAT sa, TENSOR_FLOAT sb ){ \
            cpu_template::func_name##_template( dst, a, b, sa, sb );    \
        }                                                               \
        inline void func_name( CTensor3D &dst , const CTensor3D &a, const CTensor3D &b ,TENSOR_FLOAT sa, TENSOR_FLOAT sb ){ \
            cpu_template::func_name##_template( dst, a, b, sa, sb );    \
        }                                                               \
        inline void func_name( CTensor4D &dst , const CTensor4D &a, const CTensor4D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb ){ \
            cpu_template::func_name##_template( dst, a, b, sa, sb );    \
        }                                                               \

    };
    // interface funtions 
    namespace tensor{        
        // alloc_spaceate space for given tensor
        APEX_CPU_USE_TEMPLATE_A( alloc_space )
        APEX_CPU_USE_TEMPLATE_A( free_space  )
        APEX_CPU_USE_TEMPLATE_B( fill, TENSOR_FLOAT val, val,  )
        APEX_CPU_USE_TEMPLATE_B( sample_gaussian, TENSOR_FLOAT sd , sd,  )
        APEX_CPU_USE_TEMPLATE_B( save_to_file   , FILE *dst_file, dst_file, const )
		APEX_CPU_USE_TEMPLATE_B( load_from_file , FILE *src_file, src_file, )
        APEX_CPU_USE_TEMPLATE_C( add )
        APEX_CPU_USE_TEMPLATE_C( sub )
        APEX_CPU_USE_TEMPLATE_C( mul )
        
        APEX_CPU_USE_TEMPLATE_D( add, TENSOR_FLOAT val, val )
        APEX_CPU_USE_TEMPLATE_D( mul, TENSOR_FLOAT val, val )
        APEX_CPU_USE_TEMPLATE_D( sadd__mul, TENSOR_FLOAT val, val )
        APEX_CPU_USE_TEMPLATE_D( sample_gaussian, TENSOR_FLOAT sd, sd )
        APEX_CPU_USE_TEMPLATE_E( sigmoid )
        APEX_CPU_USE_TEMPLATE_E( sample_binary )
        APEX_CPU_USE_TEMPLATE_E( copy )
        APEX_CPU_USE_TEMPLATE_F( scale_add )        
        APEX_CPU_USE_TEMPLATE_F( sadd__scale_add )        

        APEX_CPU_USE_TEMPLATE_C( sadd__abs_err )
        APEX_CPU_USE_TEMPLATE_C( sadd__abs_err_rel )
        APEX_CPU_USE_TEMPLATE_C( sadd__abs_err_relT )
    };

	namespace tensor{
    // definition of macros
#define APEX_CPU_SUPPORT_DOT_1D(func_name,op1,op2)                          \
        inline void func_name( CTensor1D &dst, const CTensor1D &srca, const CTensor2D &srcb ){ \
            for( int i = 0; i < dst.x_max; i ++ ){                      \
	    		TENSOR_FLOAT tmp = 0;                                   \
				for( int j = 0; j < srca.x_max; j ++ )                  \
					op1;                                                \
				dst[i] op2 tmp;                                         \
	    	}                                                           \
		}                                                               \
	
#define APEX_CPU_SUPPORT_DOT_2D(func_name)                                  \
        inline void func_name( CTensor2D &dst , const CTensor2D &srca, const CTensor2D &srcb ){ \
            for( int i = 0; i < cpu_template::num_line( dst ); i ++ ){                \
                CTensor1D dd = dst[i];                                  \
                func_name( dd, srca[i], srcb );                         \
            }                                                           \
        }                                                               \

#define APEX_CPU_SUPPORT_DOT_LT_1D(func_name,op)                            \
        inline void func_name( CTensor2D &dst, const CTensor1D &srca, const CTensor1D &srcb ){ \
            for( int i = 0; i < cpu_template::num_line( dst ); i ++ ){                \
                for( int j = 0; j < dst.x_max; j ++ )                   \
                    dst[i][j] op srca[i] * srcb[j];                     \
            }                                                           \
		}                                                               \

#define APEX_CPU_SUPPORT_DOT_LT_2D(func_name,op)                        \
        inline void func_name( CTensor2D &dst, const CTensor2D &srca, const CTensor2D &srcb ){ \
            for( int i = 0; i < cpu_template::num_line( dst ); i ++){   \
				for( int j = 0; j < dst.x_max; j ++) {                  \
					TENSOR_FLOAT tmp = 0;                               \
					for( int k = 0; k < cpu_template::num_line( srca ); ++k) \
						tmp += srca[k][i] * srcb[k][j];                 \
					dst[i][j] op tmp;                                   \
				}                                                       \
			}                                                           \
		}                                                               \

    };

	namespace tensor{
        //support dot operation
		APEX_CPU_SUPPORT_DOT_1D( dot_org      , tmp += srca[j]*srcb[j][i] , =  )
		APEX_CPU_SUPPORT_DOT_1D( sadd__dot_org, tmp += srca[j]*srcb[j][i] , += )
		APEX_CPU_SUPPORT_DOT_1D( ssub__dot_org, tmp += srca[j]*srcb[j][i] , -= )
        
        APEX_CPU_SUPPORT_DOT_2D( dot_org )                          
        APEX_CPU_SUPPORT_DOT_2D( sadd__dot_org )
		APEX_CPU_SUPPORT_DOT_2D( ssub__dot_org )

   		APEX_CPU_SUPPORT_DOT_1D( dot_rt_org      , tmp += srca[j]*srcb[i][j] , =  )
		APEX_CPU_SUPPORT_DOT_1D( sadd__dot_rt_org, tmp += srca[j]*srcb[i][j] , += )
		APEX_CPU_SUPPORT_DOT_1D( ssub__dot_rt_org, tmp += srca[j]*srcb[i][j] , -= )

        APEX_CPU_SUPPORT_DOT_2D( dot_rt_org )                          
        APEX_CPU_SUPPORT_DOT_2D( sadd__dot_rt_org )
		APEX_CPU_SUPPORT_DOT_2D( ssub__dot_rt_org )

		APEX_CPU_SUPPORT_DOT_LT_1D( dot_lt_org      , =  )
		APEX_CPU_SUPPORT_DOT_LT_1D( sadd__dot_lt_org, += )
		APEX_CPU_SUPPORT_DOT_LT_1D( ssub__dot_lt_org, -= )

		APEX_CPU_SUPPORT_DOT_LT_2D( dot_lt_org      , =  )
		APEX_CPU_SUPPORT_DOT_LT_2D( sadd__dot_lt_org, += )
		APEX_CPU_SUPPORT_DOT_LT_2D( ssub__dot_lt_org, -= )
 
    };   
    /* 
       Use BLAS to speed up matrix computation 
     */
// matrix multiplication that can be optimized using BLAS    


#if __APEX_TENSOR_USE_BLAS__
    namespace tensor{
        inline void dot_blas( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemv( CblasRowMajor, CblasTrans, b.y_max, b.x_max, 1.0 , b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 0.0, dst.elem, 1 );
#else
            cblas_sgemv( CblasRowMajor, CblasTrans, b.y_max, b.x_max, 1.0f, b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 0.0f,dst.elem, 1 );
#endif
        }
        inline void sadd__dot_blas( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemv( CblasRowMajor, CblasTrans, b.y_max, b.x_max, 1.0 , b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0 ,dst.elem, 1 );
#else
            cblas_sgemv( CblasRowMajor, CblasTrans, b.y_max, b.x_max, 1.0f, b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0f,dst.elem, 1 );
#endif
        }
        inline void ssub__dot_blas( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemv( CblasRowMajor, CblasTrans, b.y_max, b.x_max, -1.0 , b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0 , dst.elem, 1 );
#else
            cblas_sgemv( CblasRowMajor, CblasTrans, b.y_max, b.x_max, -1.0f, b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0f, dst.elem, 1 );
#endif
        }
        inline void dot_blas( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                         a.y_max, b.x_max, a.x_max, 1.0 , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                         a.y_max, b.x_max, a.x_max, 1.0f , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0f, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        inline void sadd__dot_blas( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                         a.y_max, b.x_max, a.x_max, 1.0 , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                         a.y_max, b.x_max, a.x_max, 1.0f , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0f, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        inline void ssub__dot_blas( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                         a.y_max, b.x_max, a.x_max, -1.0 , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                         a.y_max, b.x_max, a.x_max, -1.0f , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0f, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }

        inline void dot_rt_blas( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemv( CblasRowMajor, CblasNoTrans, b.y_max, b.x_max, 1.0 , b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 0.0, dst.elem, 1 );
#else
            cblas_sgemv( CblasRowMajor, CblasNoTrans, b.y_max, b.x_max, 1.0f, b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 0.0f,dst.elem, 1 );
#endif
        }

        inline void sadd__dot_rt_blas( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemv( CblasRowMajor, CblasNoTrans, b.y_max, b.x_max, 1.0 , b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0, dst.elem, 1 );
#else
            cblas_sgemv( CblasRowMajor, CblasNoTrans, b.y_max, b.x_max, 1.0f, b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0f,dst.elem, 1 );
#endif
        }
        inline void ssub__dot_rt_blas( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemv( CblasRowMajor, CblasNoTrans, b.y_max, b.x_max, -1.0 , b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0, dst.elem, 1 );
#else
            cblas_sgemv( CblasRowMajor, CblasNoTrans, b.y_max, b.x_max, -1.0f, b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), a.elem, 1, 1.0f,dst.elem, 1 );
#endif
        }
        inline void dot_rt_blas( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasTrans, 
                         a.y_max, b.y_max, a.x_max, 1.0 , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasTrans, 
                         a.y_max, b.y_max, a.x_max, 1.0f , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0f, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        inline void sadd__dot_rt_blas( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasTrans, 
                         a.y_max, b.y_max, a.x_max, 1.0 , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasTrans, 
                         a.y_max, b.y_max, a.x_max, 1.0f , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0f, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        inline void ssub__dot_rt_blas( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasTrans, 
                         a.y_max, b.y_max, a.x_max, -1.0 , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasTrans, 
                         a.y_max, b.y_max, a.x_max, -1.0f , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0f, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }

        inline void dot_lt_blas( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemm( CblasRowMajor, CblasTrans, CblasNoTrans, 
                         a.x_max, b.x_max, 1, 1.0 , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sgemm( CblasRowMajor, CblasTrans, CblasNoTrans, 
                         a.x_max, b.x_max, 1, 1.0f , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0f, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }                
        inline void sadd__dot_lt_blas( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dger( CblasRowMajor, a.x_max, b.x_max, 1.0 , a.elem, 1, b.elem, 1, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sger( CblasRowMajor, a.x_max, b.x_max, 1.0f, a.elem, 1, b.elem, 1, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        inline void ssub__dot_lt_blas( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dger( CblasRowMajor, a.x_max, b.x_max, -1.0 , a.elem, 1, b.elem, 1, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sger( CblasRowMajor, a.x_max, b.x_max, -1.0f, a.elem, 1, b.elem, 1, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }

        inline void dot_lt_blas( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemm( CblasRowMajor, CblasTrans, CblasNoTrans, 
                         a.x_max, b.x_max, b.y_max, 1.0 , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sgemm( CblasRowMajor, CblasTrans, CblasNoTrans, 
                         a.x_max, b.x_max, b.y_max, 1.0f , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         0.0f, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        inline void sadd__dot_lt_blas( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemm( CblasRowMajor, CblasTrans, CblasNoTrans, 
                         a.x_max, b.x_max, b.y_max, 1.0 , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sgemm( CblasRowMajor, CblasTrans, CblasNoTrans, 
                         a.x_max, b.x_max, b.y_max, 1.0f , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0f, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
        inline void ssub__dot_lt_blas( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
#if __APEX_TENSOR_DOUBLE_PRECISION__
            cblas_dgemm( CblasRowMajor, CblasTrans, CblasNoTrans, 
                         a.x_max, b.x_max, b.y_max, -1.0 , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#else
            cblas_sgemm( CblasRowMajor, CblasTrans, CblasNoTrans, 
                         a.x_max, b.x_max, b.y_max, -1.0f , 
                         a.elem, a.pitch/(sizeof(TENSOR_FLOAT)), 
                         b.elem, b.pitch/(sizeof(TENSOR_FLOAT)), 
                         1.0f, dst.elem, dst.pitch/(sizeof(TENSOR_FLOAT)) );
#endif
        }
    };
#endif
    /*----------------------------------------------------------------------*/
#if __APEX_TENSOR_USE_BLAS__
    namespace tensor{
        inline void dot( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            dot_blas( dst, a, b );
        }
        inline void dot( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            dot_blas( dst, a, b );
        }
        inline void sadd__dot( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            sadd__dot_blas( dst, a, b );
        }
        inline void sadd__dot( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            sadd__dot_blas( dst, a, b );
        }
        inline void ssub__dot( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            ssub__dot_blas( dst, a, b );
        }
        inline void ssub__dot( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            ssub__dot_blas( dst, a, b );
        }
        
        inline void dot_rt( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            dot_rt_blas( dst, a, b );
        }
        inline void dot_rt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            dot_rt_blas( dst, a, b );
        }
        inline void sadd__dot_rt( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            sadd__dot_rt_blas( dst, a, b );
        }
        inline void sadd__dot_rt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            sadd__dot_rt_blas( dst, a, b );
        }
        inline void ssub__dot_rt( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            ssub__dot_rt_blas( dst, a, b );
        }
        inline void ssub__dot_rt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            ssub__dot_rt_blas( dst, a, b );
        }
        
        inline void dot_lt( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b ){
            dot_lt_blas( dst, a, b );
        }
        inline void dot_lt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            dot_lt_blas( dst, a, b );
        }
        inline void sadd__dot_lt( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b ){
            sadd__dot_lt_blas( dst, a, b );
        }
        inline void sadd__dot_lt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            sadd__dot_lt_blas( dst, a, b );
        }
        inline void ssub__dot_lt( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b ){
            ssub__dot_lt_blas( dst, a, b );
        }
        inline void ssub__dot_lt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            ssub__dot_lt_blas( dst, a, b );
        }
    };
#else
    namespace tensor{
        inline void dot( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            dot_org( dst, a, b );
        }
        inline void dot( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            dot_org( dst, a, b );
        }
        inline void sadd__dot( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            sadd__dot_org( dst, a, b );
        }
        inline void sadd__dot( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            sadd__dot_org( dst, a, b );
        }
        inline void ssub__dot( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            ssub__dot_org( dst, a, b );
        }
        inline void ssub__dot( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            ssub__dot_org( dst, a, b );
        }

        inline void dot_rt( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            dot_rt_org( dst, a, b );
        }
        inline void dot_rt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            dot_rt_org( dst, a, b );
        }
        inline void sadd__dot_rt( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            sadd__dot_rt_org( dst, a, b );
        }
        inline void sadd__dot_rt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            sadd__dot_rt_org( dst, a, b );
        }
        inline void ssub__dot_rt( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            ssub__dot_rt_org( dst, a, b );
        }
        inline void ssub__dot_rt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            ssub__dot_rt_org( dst, a, b );
        }
        
        inline void dot_lt( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b ){
            dot_lt_org( dst, a, b );
        }
        inline void dot_lt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            dot_lt_org( dst, a, b );
        }
        inline void sadd__dot_lt( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b ){
            sadd__dot_lt_org( dst, a, b );
        }
        inline void sadd__dot_lt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            sadd__dot_lt_org( dst, a, b );
        }
        inline void ssub__dot_lt( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b ){
            ssub__dot_lt_org( dst, a, b );
        }
        inline void ssub__dot_lt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b ){
            ssub__dot_lt_org( dst, a, b );
        }
    };
#endif

    namespace tensor{
        inline TENSOR_FLOAT sum_mul( const CTensor1D &a, const CTensor1D &b ){
            TENSOR_FLOAT ans = 0;
            for( int i = 0; i < a.x_max; i ++ )
                ans += a[ i ] * b[ i ];
            return ans;
        }        
    };

    // support for CRBM
    namespace tensor{
        namespace cpu_store_method{
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

        namespace crbm{            
            // fit the last two dimension of src into dst's size, copy the fitted part into dst
            inline void copy_fit( CTensor2D &dst, const CTensor2D &src ){
                cpu_template::copy_template( dst, src );
            }
            inline void copy_fit( CTensor3D &dst, const CTensor3D &src ){
                for( int i = 0 ; i < dst.z_max ; i ++ )
                    cpu_template::copy_template( dst[i], src[i] );                         
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
            
            inline void norm_maxpooling_2D( CTensor3D &mean, const CTensor3D &energy, int pool_size ){
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
                                if( !hit && sum > rnd ){
                                    state[y+dy][x+dx] = 1.0f; hit = true; 
                                }else{
                                    state[y+dy][x+dx] = 0.0f; 
                                }                                
                            }
                    }
            }

            inline void sample_maxpooling_2D_inner_B( CTensor2D state, const CTensor2D mean, int pool_size ){                
                for( int y = 0 ; y < state.y_max ; y += pool_size )
                    for( int x = 0 ; x < state.x_max ; x += pool_size ){
                        double sum = 0.0f;
                        for( int dy = 0 ; dy < pool_size && y+dy < state.y_max ; dy ++ )
                            for( int dx = 0 ; dx < pool_size && x+dx < state.x_max ; dx ++ )
                              sum += mean[y+dy][x+dx];
                        
                        if( apex_random::sample_binary( sum )){
                            bool hit = false;
                            double ss = 0.0;   
                            double rnd = apex_random::next_double();
                            for( int dy = 0 ; dy < pool_size && y+dy < state.y_max ; dy ++ )
                                for( int dx = 0 ; dx < pool_size && x+dx < state.x_max ; dx ++ ){
                                    ss += mean[y+dy][x+dx] / sum;
                                    if( !hit && ss > rnd ){
                                        state[y+dy][x+dx] = 1.0f; hit = true; 
                                    }else{
                                        state[y+dy][x+dx] = 0.0f; 
                                    }                                
                                }
                        }else{
                            for( int dy = 0 ; dy < pool_size && y+dy < state.y_max ; dy ++ )
                                for( int dx = 0 ; dx < pool_size && x+dx < state.x_max ; dx ++ )
                                    state[y+dy][x+dx] = 0.0f;
                        }
                    }
            }

            // sample the data using 2D maxpooling 
            inline void sample_maxpooling_2D( CTensor3D &state, const CTensor3D &mean, int pool_size ){
                for( int z = 0 ; z < state.z_max ; z ++ )
                    sample_maxpooling_2D_inner_B( state[z], mean[z], pool_size );
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
                        cpu_store_method::__store<st_method> ( dst[yy][xx] , sum );
                    }
            }

            // pool up
            inline void pool_up( CTensor3D &dst , const CTensor3D &src, int pool_size ){
                for( int z = 0 ; z < dst.z_max ; z ++ )
                    pool_up_inner<cpu_store_method::SAVE>( dst[z], src[z], pool_size );
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
                        cpu_store_method::__store<st_method>( dst[y][x] , sum );
                    }
            }
            template<int st_method>
            inline void conv2_r_valid_inner( CTensor3D &dst, const CTensor3D &a, const CTensor4D &filter ){
                if( st_method == cpu_store_method::SAVE ){                    
                    for( int h = 0 ; h < dst.z_max ; h ++ )
                        conv2_r_valid_inner<cpu_store_method::SAVE>( dst[h], a[0], filter[0][h] ); 

                    for( int v = 1 ; v < a.z_max ; v ++ )
                        for( int h = 0 ; h < dst.z_max ; h ++ )
                            conv2_r_valid_inner<cpu_store_method::ADD>( dst[h], a[v], filter[v][h] ); 
                }
                else{
                    for( int v = 0 ; v < a.z_max ; v ++ )
                        for( int h = 0 ; h < dst.z_max ; h ++ ){
                            conv2_r_valid_inner<st_method>( dst[h], a[v], filter[v][h] ); 
                        }

                }
            }
                        
            inline void conv2_r_valid     ( CTensor3D &dst, const CTensor3D &a, const CTensor4D &filter, const CTensor1D &bias ){
                for( int h = 0 ; h < dst.z_max ; h ++ )
                    dst[ h ] = bias[ h ];
                conv2_r_valid_inner<cpu_store_method::ADD>( dst, a, filter );
            }
            
            template<int st_method>
            inline void conv2_full_inner( CTensor2D dst, const CTensor2D a, const CTensor2D filter ){
                if( st_method == cpu_store_method::SAVE ) dst = 0.0f; 

                for( int y = 0 ; y < a.y_max ; y ++ )
                    for( int x = 0 ; x < a.x_max ; x ++ )
                        for( int dy = 0 ; dy < filter.y_max ; dy ++ )
                            for( int dx = 0 ; dx < filter.x_max ; dx ++ )
                                cpu_store_method::__store<st_method>( dst[y+dy][x+dx], a[y][x]*filter[dy][dx] );
            }
            
            template<int st_method>
            inline void conv2_full_inner( CTensor3D &dst, const CTensor3D &a, const CTensor4D &filter ){
                if( st_method == cpu_store_method::SAVE ){      
                    dst = 0.0f;
                    for( int v = 0 ; v < dst.z_max ; v ++ ){
                        for( int h = 0 ; h < a.z_max ; h ++ )
                            conv2_full_inner<cpu_store_method::ADD>( dst[v], a[h], filter[v][h] ); 
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
            inline void conv2_full        ( CTensor3D &dst, const CTensor3D &a, const CTensor4D &filter, const CTensor1D &bias ){
                for( int v = 0 ; v < dst.z_max ; v ++ )
                    dst[v] = bias[v];
                conv2_full_inner<cpu_store_method::ADD>( dst , a, filter ); 
            }
            
            // convolution with big filter
            inline void sadd__conv2_r_big_filter( CTensor4D &dst, const CTensor3D &a, const CTensor3D &b ){
                for( int v = 0 ; v < a.z_max ; v ++ )
                    for( int h = 0 ; h < b.z_max ; h ++ )
                        conv2_r_valid_inner<cpu_store_method::ADD>( dst[v][h], a[v], b[h] );
            }
            inline void ssub__conv2_r_big_filter( CTensor4D &dst, const CTensor3D &a, const CTensor3D &b ){
                for( int v = 0 ; v < a.z_max ; v ++ )
                    for( int h = 0 ; h < b.z_max ; h ++ )
                        conv2_r_valid_inner<cpu_store_method::SUB>( dst[v][h], a[v], b[h] );
            }
            
            // sum over last two dimension
            inline void sadd__sum_2D( CTensor1D &dst, const CTensor3D &src ){
                for( int i = 0 ; i < dst.x_max ; i ++ )
                    dst[i] += cpu_only::sum_template( src[i] );
            }
            inline void ssub__sum_2D( CTensor1D &dst, const CTensor3D &src ){
                for( int i = 0 ; i < dst.x_max ; i ++ )
                    dst[i] -= cpu_only::sum_template( src[i] );
            }            

            inline void sum_2D    ( CTensor2D &dst, const CTensor4D &src ){
                for( int i = 0 ; i < dst.y_max ; i ++ )
                    for( int j = 0 ; j < dst.x_max ; j ++ )
                        dst[i][j] = cpu_only::sum_template( src[i][j] );
            }            
                        
            inline void sadd__scale( CTensor4D &dst, const CTensor2D &src, TENSOR_FLOAT scale_src ){
                for( int i = 0 ; i < src.y_max ; i ++ )
                    for( int j = 0 ; j < src.x_max ; j ++ ){
                        dst[i][j] += src[i][j] * scale_src;
                    }
            }
            
            inline void refill_edge_area( CTensor3D &dst, const CTensor3D &src, int edge_y_len, int edge_x_len ){
                for( int i = 0 ; i < dst.z_max ; i ++ )
                    for( int y = 0 ; y < dst.y_max ; y ++ )
                        for( int x = 0 ; x < dst.x_max ; x ++ )
                            if( y < edge_y_len || y >= dst.y_max - edge_y_len ||
                                x < edge_x_len || x >= dst.x_max - edge_x_len ){
                                    dst[i][y][x] = src[i][y][x];
                                }                            
            }
            
            // calculate information of sparse regularization
            inline void add_sparse_info( CTensor1D &sum_mf, CTensor1D &sum_mf_grad, const CTensor3D &src, int pool_size ){
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

/** support for sparse tensor */
namespace apex_tensor{
    namespace tensor{
        // allocate space for index 
        inline void alloc_space_index( CSparseIndex2D &index ){
            index.y = new int[ index.alloc_length ];
            index.x = new int[ index.alloc_length ];        
        }
        // allocate space using setting of index
        inline CTensor2DSparse alloc_space_data( CSparseIndex2D index ){
            CTensor2DSparse ts;
            ts.index = index;
            ts.elem  = new TENSOR_FLOAT[ index.alloc_length ]; 
            return ts;
        }
        // free the index space 
        inline void free_space_index( CSparseIndex2D  &index ){
            delete [] index.y;
            delete [] index.x; 
        }
        // free data space of tensor
        inline void free_space_data ( CTensor2DSparse &ts ){
            delete [] ts.elem;
        }
        // copy index from cpu to cpu
        inline void copy_index ( CSparseIndex2D &dst , const CSparseIndex2D &a ){
            memcpy( dst.y , a.y, sizeof(int) * a.length );
            memcpy( dst.x , a.x, sizeof(int) * a.length );
        }
        // copy from cpu to cpu
        inline void copy_data  ( CTensor2DSparse &dst, const CTensor2DSparse &a ){
            memcpy( dst.elem , a.elem, sizeof(TENSOR_FLOAT) * a.index.length );
        }
    };

    /** more complicated operations for tensor */
    namespace tensor{
        // dst = a + b;
        inline void add   ( CTensor2DSparse &dst , const CTensor2DSparse &a, const CTensor2DSparse &b ){
            cpu_template::check_true( dst.index == a.index && dst.index == b.index, "add::index set of sparse tensor must be same");
            for( unsigned int i = 0 ; i < dst.index.length; i ++ )
                dst.elem[i] = a.elem[i] + b.elem[i];
        }
        // dst = a - b;
        inline void sub   ( CTensor2DSparse &dst , const CTensor2DSparse &a, const CTensor2DSparse &b ){
            cpu_template::check_true( dst.index == a.index && dst.index == b.index, "add::index set of sparse tensor must be same");
            for( unsigned int i = 0 ; i < dst.index.length; i ++ )
                dst.elem[i] = a.elem[i] - b.elem[i];
        }
        
        template<int st_m>
        inline void dot_rt_template( CTensor2DSparse &dst, const CTensor2D &a, const CTensor2D &b ){
            cpu_template::check_true( a.x_max == b.x_max, "dot_rt::matrix dimension must agree");
            for( unsigned int i = 0; i < dst.index.length; i ++ ){
                int y = dst.index.y[i];
                int x = dst.index.x[i];
                TENSOR_FLOAT sum = 0;
                for( int j = 0; j < a.x_max; j ++ ){
                    sum += a[y][j] * b[x][j];
                }
                cpu_store_method::__store<st_m>( dst.elem[ i ], sum );
            }
        }
        inline void dot_rt( CTensor2DSparse &dst , const CTensor2D &a , const CTensor2D &b ){
            dot_rt_template<cpu_store_method::SAVE>( dst, a, b );
        }
        inline void sadd__dot_rt( CTensor2DSparse &dst , const CTensor2D &a, const CTensor2D &b ){
            dot_rt_template<cpu_store_method::ADD> ( dst, a, b );
        }
        
        template<int st_m>
        inline void dot_template( CTensor2D &dst , const CTensor2DSparse &W, const CTensor2D &P ){
            cpu_template::check_true( dst.x_max == P.x_max, "dot_rt::matrix dimension must agree");
            for( unsigned int i = 0; i < W.index.length; i ++ ){
                int y = W.index.y[i];
                int x = W.index.x[i];

                for( int j = 0; j < dst.x_max; j ++ ){
                    cpu_store_method::__store<st_m>( dst[y][j], P[x][j] * W.elem[i] );
                }
            }
        }

        // dst = dot( W, P )
        inline void dot( CTensor2D &dst, const CTensor2DSparse &W, const CTensor2D &P ){
            dst = 0;
            dot_template<cpu_store_method::ADD>( dst, W, P );
        }
        inline void sadd__dot ( CTensor2D &dst , const CTensor2DSparse &W, const CTensor2D &P ){
            dot_template<cpu_store_method::ADD>( dst, W, P );
        }

        template<int st_m>
        inline void dot_lt_template( CTensor2D &dst , const CTensor2DSparse &W, const CTensor2D &P ){
            cpu_template::check_true( dst.x_max == P.x_max, "dot_rt::matrix dimension must agree");
            for( unsigned int i = 0; i < W.index.length; i ++ ){
                int y = W.index.y[i];
                int x = W.index.x[i];

                for( int j = 0; j < dst.x_max; j ++ ){
                    cpu_store_method::__store<st_m>( dst[x][j], P[y][j] * W.elem[i] );
                }
            }
        }

        // dst = dot( W.T,P )
        inline void dot_lt      ( CTensor2D &dst , const CTensor2DSparse &W, const CTensor2D &P ){
            dst = 0;
            dot_lt_template<cpu_store_method::ADD>( dst, W, P );
        }        
        inline void sadd__dot_lt( CTensor2D &dst , const CTensor2DSparse &W, const CTensor2D &P ){
            dot_lt_template<cpu_store_method::ADD>( dst, W, P );
        }        
        inline void ssub__dot_lt( CTensor2D &dst , const CTensor2DSparse &W, const CTensor2D &P ){
            dot_lt_template<cpu_store_method::SUB>( dst, W, P );
        }        
    };

    namespace tensor{
        inline CTensor1DSparse create_sparse( const std::vector<int> &idx, const std::vector<TENSOR_FLOAT> &vals ){
            CTensor1DSparse sps;
            sps.index.length = (unsigned int)idx.size();
            sps.index.alloc_length = (unsigned int)idx.size();
            sps.index.x = new int[ idx.size() ];
            sps.elem    = new TENSOR_FLOAT[ idx.size() ];
            for( size_t  i = 0 ; i < idx.size() ; i ++ ){
                sps.index.x[i]   = idx[i];
                sps.elem[i]      = vals[i];
            } 
            return sps;
        }
        inline void free_space( CTensor1DSparse &sps ){
            delete[] sps.index.x;
            delete[] sps.elem;
        }

        inline void sadd__mul( CTensor1D &dst , const CTensor1DSparse &a, TENSOR_FLOAT val ){
            for( unsigned int i = 0 ; i < a.index.length ; i ++ )
                dst[ a.index.x[i] ] += a.elem[i] * val;
        }
                
        template<int st_m>
        inline void dot_template( CTensor1D &dst , const CTensor1DSparse &W, const CTensor2D &b ){
            for( int x = 0; x < b.x_max ; x ++ ){
                TENSOR_FLOAT ans = 0;
                for( unsigned int i = 0; i < W.index.length; i ++ ){
                    int y = W.index.x[ i ];
                    ans += W.elem[ i ] * b[ y ][ x ];
                }
                cpu_store_method::__store<st_m>( dst[x], ans );
            }
        }
        inline void dot   ( CTensor1D &dst, const CTensor1DSparse &a, const CTensor2D &b ){
            dot_template<cpu_store_method::SAVE>( dst, a, b );
        }
        inline void sadd__dot( CTensor1D &dst, const CTensor1DSparse &a, const CTensor2D &b ){
            dot_template<cpu_store_method::ADD>( dst, a, b );
        }
        inline void ssub__dot( CTensor1D &dst, const CTensor1DSparse &a, const CTensor2D &b ){
            dot_template<cpu_store_method::SUB>( dst, a, b );
        }
        
        template<int st_m>
        inline void dot_template_scale( CTensor1D &dst , const CTensor1DSparse &W, const CTensor2D &b, TENSOR_FLOAT scale ){
            for( int x = 0; x < b.x_max ; x ++ ){
                TENSOR_FLOAT ans = 0;
                for( unsigned int i = 0; i < W.index.length; i ++ ){
                    int y = W.index.x[ i ];
                    ans += W.elem[ i ] * b[ y ][ x ];
                }
                cpu_store_method::__store<st_m>( dst[x], ans*scale );
            }
        }
		inline void sadd__dot_scale( CTensor1D &dst, const CTensor1DSparse &a, const CTensor2D &b, TENSOR_FLOAT scale ){
            dot_template_scale<cpu_store_method::ADD>( dst, a, b, scale );
        }
        inline void ssub__dot_scale( CTensor1D &dst, const CTensor1DSparse &a, const CTensor2D &b, TENSOR_FLOAT scale ){
            dot_template_scale<cpu_store_method::SUB>( dst, a, b, scale );
        }
        
        template<int st_m>
        inline void dot_lt_template( CTensor2D &dst , const CTensor1DSparse &W, const CTensor1D &b ){
            for( unsigned int i = 0; i < W.index.length; i ++ ){
                int y = W.index.x[i];
                for( int x = 0 ; x < b.x_max ; x ++ )
                    cpu_store_method::__store<st_m>( dst[y][x], W.elem[i]*b[x] );                                
            }
        }
        inline void dot_lt      ( CTensor2D &dst, const CTensor1DSparse &a, const CTensor1D &b ){
            dst = 0;
            dot_lt_template<cpu_store_method::ADD>( dst, a, b );
        }
        inline void sadd__dot_lt( CTensor2D &dst, const CTensor1DSparse &a, const CTensor1D &b ){
            dot_lt_template<cpu_store_method::ADD>( dst, a, b );
        }
        inline void ssub__dot_lt( CTensor2D &dst, const CTensor1DSparse &a, const CTensor1D &b ){
            dot_lt_template<cpu_store_method::SUB>( dst, a, b );
        }
        
        template<int st_m>
        inline void dot_lt_scale_template( CTensor2D &dst , const CTensor1DSparse &W, const CTensor1D &b, TENSOR_FLOAT scale ){
            for( unsigned int i = 0; i < W.index.length; i ++ ){
                int y = W.index.x[i];
                for( int x = 0 ; x < b.x_max ; x ++ )
                    cpu_store_method::__store<st_m>( dst[y][x], W.elem[i]*b[x]*scale );                                
            }
        }
        inline void sadd__dot_lt_scale( CTensor2D &dst, const CTensor1DSparse &a, const CTensor1D &b, TENSOR_FLOAT scale ){
            dot_lt_scale_template<cpu_store_method::ADD>( dst, a, b, scale );
        }
        
        // dst = sum( a * b );
        inline TENSOR_FLOAT sum_mul( const CTensor1DSparse &a, const CTensor1D &b ){
            TENSOR_FLOAT ans = 0;
            for( unsigned int i = 0; i < a.index.length; i ++ )
                ans += a.elem[i] * b[ a.index.x[i] ];
            return ans;
        }        
    };
};
#endif

