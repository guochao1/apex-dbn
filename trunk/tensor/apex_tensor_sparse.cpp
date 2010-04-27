#ifndef _APEX_TENSOR_SPARSE_CPP_
#define _APEX_TENSOR_SPARSE_CPP_

#include "apex_tensor_sparse.h"
#include "../external/apex_random.h"
#include "memory.h"
//member function of CSTensor1D
namespace apex_tensor{

		inline TENSOR_FLOAT& CSTensor1D::operator[] ( int idx ){
				return this->elem[ idx ];
		}

		inline const TENSOR_FLOAT& CSTensor1D::operator[] ( int idx )const{
				return this->elem[ idx ];
		}

        inline CSTensor1D& CSTensor1D::operator =  ( const apex_op_plan::DotRTPlan<CTensor1D,CTensor2D> &val ){
				tensor::dot_rt( *this, *(val.a), *(val.b) );
				return *this;
		}
};
//member function of CSTensor2D
namespace apex_tensor{

        inline CSTensor1D CSTensor2D::operator[]( int idx ){
	 		CSTensor1D ts;
			ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*pitch);
		   	ts.pitch = pitch;
			ts.x_max = x_max;
			return ts;  
		}

        inline const CSTensor1D CSTensor2D::operator[]( int idx )const{
			CSTensor1D ts;
			ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*pitch);
		   	ts.pitch = pitch;
			ts.x_max = x_max;
			return ts;  
		}

		inline CSTensor2D& CSTensor2D::operator =  ( const apex_op_plan::AllocLikePlan<CSTensor2D> &val ){
			this->set_param( val.a->y_max, val.a->x_max );
			tensor::alloc_space( *this );
			return *this;
		}
};
namespace apex_tensor{
	namespace tensor{

		inline void sadd( CSTensor1D &dst, const CTensor1D &src){
				for( int i = 0; i < dst.x_max; ++ i ){
					dst.elem[ i ] = dst.elem[ i ] + src.elem[ dst.index[ i ] ];
				}
		}

		inline void add( CTensor2D &dst,  const CTensor2D &a,  const CSTensor2D &b ){
				for( int i = 0; i < dst.y_max; ++ i ){
					for(int j = 0; j < a.x_max; ++j){
						dst[ i ][ b.index[ j ] ] = a[ i ][ b.index[ j] ] + b[ i ][ j ];
					}
				}
		}

		inline void sub( CTensor2D &dst,  const CTensor2D &a,  const CSTensor2D &b ){
				for( int i = 0; i < dst.y_max; ++ i ){
					for(int j = 0; j < a.x_max; ++j){
						dst[ i ][ b.index[ j ] ] = a[ i ][ b.index[ j] ] - b[ i ][ j ];
					}
				}
		}

		inline void alloc_space( CSTensor2D &ts){
				ts.pitch = ts.x_max * sizeof(TENSOR_FLOAT);
				ts.index = new int[ ts.x_max * ts.y_max ];
				ts.elem	 = new TENSOR_FLOAT[ ts.x_max * ts.y_max ];
		}

		inline void free_space( CSTensor2D &ts ){
				delete[] ts.index;
				delete[] ts.elem;
		}

		inline void copy( CSTensor2D &dst, const CSTensor2D &src ){
				for( int i = 0 ; i <  dst.y_max  ; ++ i ){
					int *di = dst[ i ].index;
					int *si = src[ i ].index;
					memcpy( di, si, dst.x_max );
					TENSOR_FLOAT *de =  dst[ i ].elem;
					TENSOR_FLOAT *se =  src[ i ].elem;
					memcpy( de, se, sizeof( TENSOR_FLOAT ) * dst.x_max );
				}
		}

        inline void dot( CTensor1D &dst, const CSTensor1D &a, const CTensor2D &b ){
            for( int i = 0; i < dst.x_max; ++ i ){                      
				TENSOR_FLOAT tmp = 0;                                   
				for( int j = 0; j < a.x_max; ++ j )                  
					tmp += a[ j ]*b[ j ][ a.index[ i ] ];                                                
				dst[ a.index[ i ] ] = tmp;                                         
			}
		}

		inline void dot_rt( CSTensor1D &dst, const CTensor1D &a, const CTensor2D &b ){
			  for( int i = 0; i < dst.x_max; ++ i ){                      
				TENSOR_FLOAT tmp = 0;                                   
				for( int j = 0; j < a.x_max; ++ j )                  
					tmp += a[ dst.index[ j ] ] * b[ i ][ dst.index[ j ] ];                                                
				dst[ i ] = tmp;                                         
			  }
		}

        inline void sadd__dot( CTensor1D &dst, const CSTensor1D &a, const CTensor2D &b ){
			    for( int i = 0; i < dst.x_max; ++ i ){                      
				TENSOR_FLOAT tmp = 0;                                   
				for( int j = 0; j < a.x_max; ++ j )                  
					tmp += a[ j ] * b[ j ][ a.index[ i ] ];                                                
				dst[ a.index[ i ] ] += tmp;                                         
			}
		}

        inline void sadd__dot_lt( CTensor2D &dst, const CSTensor1D &a, const CTensor1D &b ){
            for( int i = 0; i < a.x_max; ++ i ){
                for( int j = 0; j < b.x_max; ++ j )
                    dst[ a.index[ i ] ][ j ] += a[ i ] * b[ j ];
            }
		}

        inline void ssub__dot_lt( CTensor2D &dst, const CSTensor1D &a, const CTensor1D &b ){
            for( int i = 0; i < a.x_max; ++ i ){
                for( int j = 0; j < b.x_max; ++ j )
                    dst[ a.index[ i ] ][ j ] -= a[ i ] * b[ j ];
            }
		}

		inline void sample_softmax( CSTensor2D &dst, const CSTensor2D &mean ){
			dst.scale_set( 0 );
			TENSOR_FLOAT tmp[ dst.y_max ];
			memset( tmp, 0, dst.y_max );
            for( int i = 0 ; i < dst.x_max; ++ i ){
				for( int j = 0; j < dst.y_max; ++ j )
					tmp[ j ] = j==0?mean[ j ][ i ]: tmp[ j - 1 ] + mean[ j ][ i ];
				double ran = apex_random::next_double(); 
				for( int j = 0; j < dst.y_max; ++ j )
					if( ran < tmp[ j ] ){
						dst[ j ][ i ] = 1;
						break;
					}
            }
        }
		
	};
	
	namespace cf_rbm{
		inline void normalize( TSTensor2D &soft_max ){
				TENSOR_FLOAT sumline[ soft_max.x_max ];
				memset( sumline, 0, soft_max.x_max );
				for( int i = 0; i < soft_max.y_max; ++ i){
						TSTensor1D line = soft_max[ i ];
						for( int j = 0; j < line.x_max; ++ j )
								sumline[ j ] += line[ j ];
				}
				for( int i = 0; i < soft_max.y_max; ++ i){
						TSTensor1D line = soft_max[ i ];
						for( int j = 0; j < line.x_max; ++ j )
								line[ j ] /= sumline[ j ];
				}
		}

	};

};
#endif
