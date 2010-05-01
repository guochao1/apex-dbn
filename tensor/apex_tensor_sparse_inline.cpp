/* inline functions fo sparse tensor */
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

	APEX_ADD_SUPPORT_ALLOC_LIKE_OP( CSTensor2D );
	APEX_ADD_SUPPORT_CLONE_OP     ( CSTensor2D );	
};
