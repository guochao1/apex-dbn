#ifndef _APEX_TENSOR_CPP_
#define _APEX_TENSOR_CPP_

#include "apex_tensor.h"
#include "../external/apex_random.h"

// defintiions for tensor functions 
// tqchen

namespace apex_tensor{

    // private functions used to support tensor op 
    namespace tensor{     
        inline size_t num_bytes( Tensor1D ts ){
            return ts.pitch;
        }
        
        inline size_t num_line( Tensor1D ts ){
            return 1;
        }
        
        inline size_t num_header_bytes( Tensor1D ts ){
            return sizeof(size_t)*1;
        }
        
        inline size_t num_bytes( Tensor2D ts ){
            return ts.pitch*ts.y_max;
        }
        
        inline size_t num_line( Tensor2D ts ){
            return ts.y_max;
        }
        
        inline size_t num_header_bytes( Tensor2D ts ){
            return sizeof(size_t)*2;
        }
        
        inline size_t num_bytes( Tensor3D ts ){
            return ts.pitch*ts.y_max*ts.z_max;
        }
        
        inline size_t num_line( Tensor3D ts ){
            return ts.y_max*ts.z_max;
        }
        
        inline size_t num_header_bytes( Tensor3D ts ){
            return sizeof(size_t)*3;
        }
        
        inline size_t num_bytes( Tensor4D ts ){
            return ts.pitch*ts.y_max*ts.z_max*ts.h_max;
        }
        
        inline size_t num_line( Tensor4D ts ){
            return ts.y_max*ts.z_max*ts.h_max;
        }
        
        inline size_t num_header_bytes( Tensor4D ts ){
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
            fread( &ts, num_header_bytes( ts ) , 1 , src );
            alloc_space( ts );
            
            for( size_t i = 0 ; i < num_line( ts ) ; i ++ ){
                TENSOR_FLOAT *a = get_line( ts, i );
                fread( a, sizeof( TENSOR_FLOAT ) , ts.x_max , src );        
            }
        }
        
#define APEX_ELEMENTWISE_ASSIGN_OP(func_name,param,op)            \
        inline void func_name( T &ts, param ){                    \
            for( size_t i = 0 ; i < num_line( ts ) ; i ++ ){      \
                TENSOR_FLOAT *d = get_line( ts, i );              \
                for( int j = 0 ; j < ts.x_max ; j ++ )            \
                    op;                                           \
            }                                                     \
        }                                                         \
        
#define APEX_ELEMENTWISE_UNARY_OP(func_name,param,op)               \
        inline void func_name( T &dst, const T&src, param ){        \
            for( size_t i = 0 ; i < num_line( dst ) ; i ++ ){       \
                TENSOR_FLOAT *d = get_line( dst, i );               \
                const TENSOR_FLOAT *a = get_line_const( src, i );   \
                for( int j = 0 ; j < dst.x_max ; j ++ )             \
                    op;                                             \
            }                                                       \
        }                                                           \

#define APEX_ELEMENTWISE_MAP_OP(func_name,op)                       \
        inline void func_name( T &dst, const T&src ){               \
            for( size_t i = 0 ; i < num_line( dst ) ; i ++ ){       \
                TENSOR_FLOAT *d = get_line( dst, i );               \
                const TENSOR_FLOAT *a = get_line_const( src, i );   \
                for( int j = 0 ; j < dst.x_max ; j ++ )             \
                    op;                                             \
            }                                                       \
        }                                                           \


#define APEX_ELEMENTWISE_BINARY_OP(func_name,op)                        \
        inline void func_name( T &dst, const T &srca, const T &srcb ){  \
            for( size_t i = 0 ; i < num_line( dst ) ; i ++ ){           \
                TENSOR_FLOAT *d = get_line( dst ,i );                   \
                const TENSOR_FLOAT *a = get_line_const( srca, i );      \
                const TENSOR_FLOAT *b = get_line_const( srcb, i );      \
                for( int j = 0 ; j < dst.x_max ; j ++ )                 \
                    op;                                                 \
            }                                                           \
		}                                                               \
        

        template<typename T>
        APEX_ELEMENTWISE_ASSIGN_OP ( fill_template, TENSOR_FLOAT val ,d[j] = val  );
        template<typename T>
        APEX_ELEMENTWISE_UNARY_OP ( add_template  , TENSOR_FLOAT val ,d[j] = a[j] + val  );
        template<typename T>
        APEX_ELEMENTWISE_UNARY_OP ( mul_template  , TENSOR_FLOAT val ,d[j] = a[j] * val  );
        template<typename T>
        APEX_ELEMENTWISE_MAP_OP   ( sigmoid_template      , d[j] = (TENSOR_FLOAT)(1.0/(1+exp(-a[j]))) );
        template<typename T>
        APEX_ELEMENTWISE_MAP_OP   ( sample_binary_template, d[j] = (TENSOR_FLOAT)apex_random::sample_binary( a[j] ) );
        template<typename T>
        APEX_ELEMENTWISE_BINARY_OP( add_template, d[j] = a[j]+b[j]);
        template<typename T>
        APEX_ELEMENTWISE_BINARY_OP( sub_template, d[j] = a[j]-b[j]);
        
    };


    // definition of macros
    namespace tensor{

#define APEX_USE_TEMPLATE_A(func_name)                                  \
        void func_name( Tensor1D &dst ){                                \
            func_name##_template( dst );                                \
        }                                                               \
        void func_name( Tensor2D &dst ){                                \
            func_name##_template( dst );                                \
        }                                                               \
        void func_name( Tensor3D &dst ){                                \
            func_name##_template( dst );                                \
        }                                                               \
        void func_name( Tensor4D &dst  ){                               \
            func_name##_template( dst );                                \
        }                                                               \

#define APEX_USE_TEMPLATE_B(func_name,param,arg,cc)                     \
        void func_name( cc Tensor1D &a, param ){                        \
            func_name##_template( a, arg );                             \
        }                                                               \
        void func_name( cc Tensor2D &a, param ){                        \
            func_name##_template( a, arg );                             \
        }                                                               \
        void func_name( cc Tensor3D &a, param ){                        \
            func_name##_template( a, arg );                             \
        }                                                               \
        void func_name( cc Tensor4D &a, param ){                        \
            func_name##_template( a, arg );                             \
        }                                                               \


#define APEX_USE_TEMPLATE_C(func_name)                                  \
        void func_name( Tensor1D &dst , const Tensor1D &a, const Tensor1D &b ){ \
            func_name##_template( dst, a, b );                          \
        }                                                               \
        void func_name( Tensor2D &dst , const Tensor2D &a, const Tensor2D &b ){ \
            func_name##_template( dst, a, b );                          \
        }                                                               \
        void func_name( Tensor3D &dst , const Tensor3D &a, const Tensor3D &b ){ \
            func_name##_template( dst, a, b );                          \
        }                                                               \
        void func_name( Tensor4D &dst , const Tensor4D &a, const Tensor4D &b ){ \
            func_name##_template( dst, a, b );                          \
        }                                                               \

#define APEX_USE_TEMPLATE_D(func_name,param,arg)                        \
        void func_name( Tensor1D &dst , const Tensor1D &a, param ){     \
            func_name##_template( dst, a, arg );                        \
        }                                                               \
        void func_name( Tensor2D &dst , const Tensor2D &a, param ){     \
            func_name##_template( dst, a, arg );                        \
        }                                                               \
        void func_name( Tensor3D &dst , const Tensor3D &a, param ){     \
            func_name##_template( dst, a, arg );                        \
        }                                                               \
        void func_name( Tensor4D &dst , const Tensor4D &a, param ){     \
            func_name##_template( dst, a, arg );                        \
        }                                                               \

#define APEX_USE_TEMPLATE_E(func_name)                                  \
        void func_name( Tensor1D &dst , const Tensor1D &a){             \
            func_name##_template( dst, a );                             \
        }                                                               \
        void func_name( Tensor2D &dst , const Tensor2D &a ){            \
            func_name##_template( dst, a );                             \
        }                                                               \
        void func_name( Tensor3D &dst , const Tensor3D &a ){            \
            func_name##_template( dst, a );                             \
        }                                                               \
        void func_name( Tensor4D &dst , const Tensor4D &a ){            \
            func_name##_template( dst, a );                             \
        }                                                               \

    };
    // interface funtions 
    namespace tensor{
        // alloc_spaceate space for given tensor
        APEX_USE_TEMPLATE_A( alloc_space )
        APEX_USE_TEMPLATE_A( free_space  )
        APEX_USE_TEMPLATE_B( fill, TENSOR_FLOAT val, val,  )
        APEX_USE_TEMPLATE_B( save_to_file, FILE *dst_file  , dst_file, const )
		APEX_USE_TEMPLATE_B( load_from_file, FILE *dst_file, dst_file, )
        APEX_USE_TEMPLATE_C( add )
        APEX_USE_TEMPLATE_C( sub )
        APEX_USE_TEMPLATE_D( add, TENSOR_FLOAT val, val )
        APEX_USE_TEMPLATE_D( mul, TENSOR_FLOAT val, val )
        APEX_USE_TEMPLATE_E( sigmoid )
        APEX_USE_TEMPLATE_E( sample_binary )
        
    };
};
#endif
