#ifndef _APEX_TENSOR_INLINE_CPP_
#define _APEX_TENSOR_INLINE_CPP_
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
		ts.elem = (TENSOR_FLOAT*)((char*)elem + idx*pitch);
        ts.x_max = x_max;
		return ts;  
	}
    // author: tqchen
    inline const Tensor1D Tensor2D::operator[]( int idx )const{
		Tensor1D ts;
		ts.elem = (TENSOR_FLOAT*)((char*)elem + idx*pitch);
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
    
};
#endif

