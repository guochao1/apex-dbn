#ifndef _APEX_TENSOR_SPARSE_H_
#define _APEX_TENSOR_SPARSE_H_

#include "apex_op_plan.h"
#include "apex_tensor.h"

namespace apex_tensor{
	struct CSTensor1D{
		int				x_max;
		unsigned int 	pitch;
		int				*index;
		TENSOR_FLOAT	*elem;
		CSTensor1D(){}
		CSTensor1D( int x_max ){
			set_param( x_max );
		}
		// set the parameter of current data
		inline void set_param( int x_max ){
			this->x_max = x_max;
		}
		inline TENSOR_FLOAT& operator[]  ( int idx );
		inline const TENSOR_FLOAT& operator[]  ( int idx )const;
        inline CSTensor1D& operator =  ( const apex_op_plan::DotRTPlan<CTensor1D,CTensor2D> &val );        
	};
	
	struct CSTensor2D{
		int				x_max, y_max;
		unsigned int	pitch;
		int				*index;
		TENSOR_FLOAT	*elem;
		CSTensor2D(){}
		CSTensor2D( int y_max, int x_max ){
			set_param( y_max, x_max );
		}
		// set the parameter of current data
		inline void set_param( int y_max, int x_max ){
			this->x_max = x_max;
			this->y_max = y_max;
		}
        // BUG here, need modification
        //		inline void scale_set( TENSOR_FLOAT val ){
        //			memset( this->elem, val, x_max * y_max  );
		//}

        inline       CSTensor1D operator[]( int idx );
        inline const CSTensor1D operator[]( int idx )const;
		inline CSTensor2D& operator =  ( const apex_op_plan::AllocLikePlan<CSTensor2D> &val );
	};

	namespace tensor{
        // dst += b
		void sadd( CSTensor1D &dst, const CTensor1D &b );
        // dst = a+b 
		void add ( CTensor2D &dst , const CTensor2D &a, const CSTensor2D &b );
        // dst = a-b; 
		void sub ( CTensor2D &dst , const CTensor2D &a, const CSTensor2D &b );
        // allocate space for ts
		void alloc_space( CSTensor2D &ts);
        // free space for ts
		void free_space ( CSTensor2D &ts );
        // copy src to dsr
		void copy( CSTensor2D &dst, const CSTensor2D &src );
	};

	namespace tensor{
        // dst = dot(a,b)
        void dot         ( CTensor1D &dst  , const CSTensor1D &a, const CTensor2D &b );    
        // dst = dot(a,b.T)
		void dot_rt		 ( CSTensor1D &dst , const CTensor1D  &a, const CTensor2D &b );
        // dst+= dot(a,b)
        void sadd__dot   ( CTensor1D  &dst , const CSTensor1D &a, const CTensor2D &b );    
        // dst+= dot(a.T,b)
        void sadd__dot_lt( CTensor2D  &dst , const CSTensor1D &a, const CTensor1D &b );    
        // dst-= dot(a.T,b)
        void ssub__dot_lt( CTensor2D  &dst , const CSTensor1D &a, const CTensor1D &b );    
	};
    
    namespace tensor{
        // special support for cf_rbm
        namespace cf_rbm{
            // normalize the data by softmax
            void norm_softmax( CSTensor2D &soft_max );
            // sample  data by soft_max
            void sample_softmax( CSTensor2D &dst, const CSTensor2D &mean );
        };
    };
};

#include "apex_tensor_sparse_inline.cpp"
#endif
