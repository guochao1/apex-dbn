#ifndef _APEX_TENSOR_SPARSE_CPP_
#define _APEX_TENSOR_SPARSE_CPP_

#include "apex_tensor_sparse.h"
#include "../external/apex_random.h"
#include "memory.h"
#include <iostream>

using namespace std;
//member function of CSTensor1D
namespace apex_tensor{
	namespace tensor{

		void sadd( CSTensor1D &dst, const CTensor1D &src){
            for( int i = 0; i < dst.x_max; ++ i ){
                dst.elem[ i ] = dst.elem[ i ] + src.elem[ dst.index[ i ] ];
            }
		}
        
		void add( CTensor2D &dst,  const CTensor2D &a,  const CSTensor2D &b ){
            for( int i = 0; i < b.y_max; ++ i ){
                for(int j = 0; j < b.x_max; ++j)
						dst[ i ][ b.index[ j ] ] = a[ i ][ b.index[ j ] ] + b[ i ][ j ];
            }
		}
        
		void sub( CTensor2D &dst,  const CTensor2D &a,  const CSTensor2D &b ){
            for( int i = 0; i < b.y_max; ++ i ){
                for(int j = 0; j < b.x_max; ++j)
                    	dst[ i ][ b.index[ j ] ] = a[ i ][ b.index[ j ] ] - b[ i ][ j ];
            }
		}
        
		void alloc_space( CSTensor2D &ts ){
            ts.pitch = ts.x_max * sizeof(TENSOR_FLOAT);
            ts.index = new int[ ts.x_max ];
            ts.elem	 = new TENSOR_FLOAT[ ts.x_max * ts.y_max ];
		}
        
		void free_space( CSTensor2D &ts ){
				delete[] ts.index;
				delete[] ts.elem;
		}
        
		void copy( CSTensor2D &dst, const CSTensor2D &src ){
			int *di = dst.index;
			int *si = src.index;
			memcpy( di, si, src.x_max * sizeof(int) );
			for( int i = 0 ; i <  dst.y_max  ; ++ i ){
				TENSOR_FLOAT *de =  dst[ i ].elem;
                TENSOR_FLOAT *se =  src[ i ].elem;
                memcpy( de, se, src.pitch );
            }
		}
        
        void dot( CTensor1D &dst, const CSTensor1D &a, const CTensor2D &b ){
            for( int i = 0; i < dst.x_max; ++ i ){                      
				TENSOR_FLOAT tmp = 0;                                   
				for( int j = 0; j < a.x_max; ++ j )                  
					tmp += a[ j ]*b[ j ][ a.index[ i ] ];                                                
				dst[ a.index[ i ] ] = tmp;                                         
			}
		}
        
		void dot_rt( CSTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
            for( int i = 0; i < dst.x_max; ++ i ){                      
				TENSOR_FLOAT tmp = 0;                                   
				for( int j = 0; j < dst.x_max; ++ j ){ 
					tmp += a[ dst.index[ j ] ] * b[ i ][ dst.index[ j ] ];                                                
				}
				dst[ i ] = tmp;                                         
            }
		}
        
        void sadd__dot( CTensor1D &dst, const CSTensor1D &a, const CTensor2D &b ){
            for( int i = 0; i < b.x_max; ++ i ){                      
				TENSOR_FLOAT tmp = 0;                                   
				for( int j = 0; j < a.x_max; ++ j )                  
					tmp += a[ j ] * b[ a.index[ j ] ][ i ];                                                
				dst[ i ] += tmp;                                         
			}
		}

        void sadd__dot_lt( CTensor2D &dst, const CSTensor1D &a, const CTensor1D &b ){
            for( int i = 0; i < a.x_max; ++ i ){
                for( int j = 0; j < b.x_max; ++ j )
                    dst[ a.index[ i ] ][ j ] += a[ i ] * b[ j ];
            }
		}

        void ssub__dot_lt( CTensor2D &dst, const CSTensor1D &a, const CTensor1D &b ){
            for( int i = 0; i < a.x_max; ++ i ){
                for( int j = 0; j < b.x_max; ++ j )
                    dst[ a.index[ i ] ][ j ] -= a[ i ] * b[ j ];
            }
		}		
	};

	namespace tensor{
        namespace cf_rbm{
            void norm_softmax( TSTensor2D &soft_max ){
				for( int i = 0; i < soft_max.x_max; ++ i){
						int sum = 0;
					for( int j = 0; j < soft_max.y_max; ++ j )
							sum ++;
					for( int j = 0; j < soft_max.y_max; ++ j)
							soft_max[ j ][ i ] /= sum;
				}
            }
            
            // tmp may not be needed
            void sample_softmax( CSTensor2D &dst, const CSTensor2D &mean ){
                for( int i = 0 ; i < dst.x_max; ++ i ){
                    double ran = apex_random::next_double(); 
					bool flag = false;
                    for( int j = 0; j < dst.y_max; ++ j ){
						if(flag){
							dst[ j ][ i ] = 0;
							continue;
						}
						ran -= mean[ j ][ i ];
						if(ran <= 0){
							dst[ j ][ i ] = 1;
						}
					}
                }
            }
        };
	};
    
};
#endif
