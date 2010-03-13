#ifndef _APEX_TENSOR_INLINE_CPP_
#define _APEX_TENSOR_INLINE_CPP_

#include "apex_tensor.h"
/* inline functions for tensor */
namespace apex_tensor{

    // author: tqchen
    inline TENSOR_FLOAT& Tensor1D::operator[]( int idx ){
        return elem[idx];
    }
    
    // author: tqchen
    inline const TENSOR_FLOAT& Tensor1D::operator[]( int idx )const{
        return elem[idx];
    }    

    
    // author: tqchen
    inline Tensor1D Tensor2D::operator[]( int idx ){
        Tensor1D ts;
		ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*pitch);
        ts.pitch = pitch;
        ts.x_max = x_max;
		return ts;  
	}
    
    // author: tqchen
    inline const Tensor1D Tensor2D::operator[]( int idx )const{
		Tensor1D ts;
		ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*pitch);
        ts.pitch = pitch;
        ts.x_max = x_max;
		return ts;  
	}
    
    // author: tqchen
    inline Tensor2D Tensor3D::operator[]( int idx ){
        Tensor2D ts;
        ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*y_max*pitch);
		ts.pitch = pitch;
		ts.x_max = x_max;
        ts.y_max = y_max;
        return ts;
    }
    
    // author: tqchen
    inline const Tensor2D Tensor3D::operator[]( int idx )const{
        Tensor2D ts;
        ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*y_max*pitch);
		ts.pitch = pitch;
		ts.x_max = x_max;
        ts.y_max = y_max;
        return ts;
    }    
    
    // author: tqchen
    inline Tensor3D Tensor4D::operator[]( int idx ){
        Tensor3D ts;
        ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*z_max*y_max*pitch);
		ts.pitch = pitch;
		ts.x_max = x_max;
        ts.y_max = y_max;
        ts.z_max = z_max;
        return ts;
    }

    // author: tqchen
    inline const Tensor3D Tensor4D::operator[]( int idx )const{
        Tensor3D ts;
        ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*z_max*y_max*pitch);
		ts.pitch = pitch;
		ts.x_max = x_max;
        ts.y_max = y_max;
        ts.z_max = z_max;
        return ts;
    }    

#define APEX_ASSIGN_FUNC_TO_OP_A( opname, func_name, param, arg )       \
    inline Tensor1D& Tensor1D::opname ( param ){                        \
        func_name( *this,arg );                                         \
        return *this;                                                   \
    }                                                                   \
    inline Tensor2D& Tensor2D::opname ( param ){                        \
        func_name( *this,arg );                                         \
        return *this;                                                   \
    }                                                                   \
    inline Tensor3D& Tensor3D::opname ( param ){                        \
        func_name( *this,arg );                                         \
        return *this;                                                   \
    }                                                                   \
    inline Tensor4D& Tensor4D::opname ( param ){                        \
        func_name( *this,arg );                                         \
        return *this;                                                   \
    }                                                                   \

#define APEX_ASSIGN_FUNC_TO_OP_B( opname, func_name, param, arg )       \
    inline Tensor1D& Tensor1D::opname ( param ){                        \
        func_name( *this,*this,arg );                                   \
        return *this;                                                   \
    }                                                                   \
    inline Tensor2D& Tensor2D::opname ( param ){                        \
        func_name( *this,*this,arg );                                   \
        return *this;                                                   \
    }                                                                   \
    inline Tensor3D& Tensor3D::opname ( param ){                        \
        func_name( *this,*this,arg );                                   \
        return *this;                                                   \
    }                                                                   \
    inline Tensor4D& Tensor4D::opname ( param ){                        \
        func_name( *this,*this,arg );                                   \
        return *this;                                                   \
    }                                                                   \

#define APEX_ASSIGN_FUNC_TO_OP_C( opname, func_name )                   \
    inline Tensor1D& Tensor1D::opname ( const Tensor1D &b ){            \
        func_name( *this,*this,b );                                     \
        return *this;                                                   \
    }                                                                   \
    inline Tensor2D& Tensor2D::opname ( const Tensor2D &b ){            \
        func_name( *this,*this,b );                                     \
        return *this;                                                   \
    }                                                                   \
    inline Tensor3D& Tensor3D::opname ( const Tensor3D &b ){            \
        func_name( *this,*this,b );                                     \
        return *this;                                                   \
    }                                                                   \
    inline Tensor4D& Tensor4D::opname ( const Tensor4D &b ){            \
        func_name( *this,*this,b );                                     \
        return *this;                                                   \
    }                                                                   \

    APEX_ASSIGN_FUNC_TO_OP_A( operator=  , tensor::fill, TENSOR_FLOAT val, val );  
    APEX_ASSIGN_FUNC_TO_OP_B( operator+= , tensor::add , TENSOR_FLOAT val, val );  
    APEX_ASSIGN_FUNC_TO_OP_B( operator*= , tensor::mul , TENSOR_FLOAT val, val );  
    APEX_ASSIGN_FUNC_TO_OP_C( operator+= , tensor::add );  
    APEX_ASSIGN_FUNC_TO_OP_C( operator-= , tensor::sub );  


};
#endif

