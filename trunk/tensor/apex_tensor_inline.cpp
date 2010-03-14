/* inline functions for tensor */
namespace apex_tensor{    
    // author: tqchen
    inline TT1D TT2D::operator[]( int idx ){
        TT1D ts;
		ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*pitch);
        ts.pitch = pitch;
        ts.x_max = x_max;
		return ts;  
	}
    
    // author: tqchen
    inline const TT1D TT2D::operator[]( int idx )const{
		TT1D ts;
		ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*pitch);
        ts.pitch = pitch;
        ts.x_max = x_max;
		return ts;  
	}
    
    // author: tqchen
    inline TT2D TT3D::operator[]( int idx ){
        TT2D ts;
        ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*y_max*pitch);
		ts.pitch = pitch;
		ts.x_max = x_max;
        ts.y_max = y_max;
        return ts;
    }
    
    // author: tqchen
    inline const TT2D TT3D::operator[]( int idx )const{
        TT2D ts;
        ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*y_max*pitch);
		ts.pitch = pitch;
		ts.x_max = x_max;
        ts.y_max = y_max;
        return ts;
    }    
    
    // author: tqchen
    inline TT3D TT4D::operator[]( int idx ){
        TT3D ts;
        ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*z_max*y_max*pitch);
		ts.pitch = pitch;
		ts.x_max = x_max;
        ts.y_max = y_max;
        ts.z_max = z_max;
        return ts;
    }

    // author: tqchen
    inline const TT3D TT4D::operator[]( int idx )const{
        TT3D ts;
        ts.elem  = (TENSOR_FLOAT*)((char*)elem + idx*z_max*y_max*pitch);
		ts.pitch = pitch;
		ts.x_max = x_max;
        ts.y_max = y_max;
        ts.z_max = z_max;
        return ts;
    }    

#define APEX_ASSIGN_FUNC_TO_OP_A( opname, func_name, param, arg )       \
    inline TT1D& TT1D::opname ( param ){                                \
        func_name( *this,arg );                                         \
        return *this;                                                   \
    }                                                                   \
    inline TT2D& TT2D::opname ( param ){                                \
        func_name( *this,arg );                                         \
        return *this;                                                   \
    }                                                                   \
    inline TT3D& TT3D::opname ( param ){                                \
        func_name( *this,arg );                                         \
        return *this;                                                   \
    }                                                                   \
    inline TT4D& TT4D::opname ( param ){                                \
        func_name( *this,arg );                                         \
        return *this;                                                   \
    }                                                                   \

#define APEX_ASSIGN_FUNC_TO_OP_B( opname, func_name, param, arg )       \
    inline TT1D& TT1D::opname ( param ){                                \
        func_name( *this,*this,arg );                                   \
        return *this;                                                   \
    }                                                                   \
    inline TT2D& TT2D::opname ( param ){                                \
        func_name( *this,*this,arg );                                   \
        return *this;                                                   \
    }                                                                   \
    inline TT3D& TT3D::opname ( param ){                                \
        func_name( *this,*this,arg );                                   \
        return *this;                                                   \
    }                                                                   \
    inline TT4D& TT4D::opname ( param ){                                \
        func_name( *this,*this,arg );                                   \
        return *this;                                                   \
    }                                                                   \
    
#define APEX_ASSIGN_FUNC_TO_OP_C( opname, func_name )                   \
    inline TT1D& TT1D::opname ( const TT1D &b ){                        \
        func_name( *this,*this,b );                                     \
        return *this;                                                   \
    }                                                                   \
    inline TT2D& TT2D::opname ( const TT2D &b ){                        \
        func_name( *this,*this,b );                                     \
        return *this;                                                   \
    }                                                                   \
    inline TT3D& TT3D::opname ( const TT3D &b ){                        \
        func_name( *this,*this,b );                                     \
        return *this;                                                   \
    }                                                                   \
    inline TT4D& TT4D::opname ( const TT4D &b ){                        \
        func_name( *this,*this,b );                                     \
        return *this;                                                   \
    }                                                                   \

    APEX_ASSIGN_FUNC_TO_OP_A( operator=  , tensor::fill, TENSOR_FLOAT val, val );  
    APEX_ASSIGN_FUNC_TO_OP_B( operator+= , tensor::add , TENSOR_FLOAT val, val );  
    APEX_ASSIGN_FUNC_TO_OP_B( operator*= , tensor::mul , TENSOR_FLOAT val, val );  
    APEX_ASSIGN_FUNC_TO_OP_C( operator+= , tensor::add );  
    APEX_ASSIGN_FUNC_TO_OP_C( operator-= , tensor::sub );  

#undef APEX_ASSIGN_FUNC_TO_OP_A
#undef APEX_ASSIGN_FUNC_TO_OP_B
#undef APEX_ASSIGN_FUNC_TO_OP_C

    // more operator support    
    // scale add
    // expand the macro to all types
#define APEX_EXPAND(mac)                                                \
    mac(TT1D)                                                           \
    mac(TT2D)                                                           \
    mac(TT3D)                                                           \
    mac(TT4D)                                                           \

#define APEX_EXPAND2(mac2)                                              \
    mac2(TT1D,TENSOR_FLOAT)                                             \
    mac2(TT2D,TENSOR_FLOAT)                                             \
    mac2(TT3D,TENSOR_FLOAT)                                             \
    mac2(TT4D,TENSOR_FLOAT)                                             \


#define APEX_EVAL_SCALE_ADD_PLAN(T)                                     \
    inline T& T::operator= ( const apex_op_plan::ScaleAddPlan<T,TENSOR_FLOAT> &val ){ \
        tensor::scale_add( *this, *(val.a),*(val.b), val.sa,val.sb);    \
        return *this;                                                   \
    }                                                                   \


#define APEX_EVAL_SCALE_PLAN(T)                                         \
    inline T& T::operator= ( const apex_op_plan::ScalePlan<T,TENSOR_FLOAT> &val ){ \
        tensor::mul( *this, *(val.a), val.scale );                      \
        return *this;                                                   \
    }                                                                   \

#define APEX_EVAL_ADD_PLAN(T)                                           \
    inline T& T::operator= ( const apex_op_plan::AddPlan<T> &val ){     \
        tensor::add( *this, *(val.a), *(val.b) );                       \
        return *this;                                                   \
    }                                                                   \

#define APEX_EVAL_DOT_PLAN(T,TA,TB)                                     \
    inline T& T::operator= ( const apex_op_plan::DotPlan<TA,TB> &val ){ \
        tensor::dot( *this, *(val.a), *(val.b) );                       \
        return *this;                                                   \
    }                                                                   \
    
    
    APEX_EXPAND(  APEX_EVAL_ADD_PLAN )
    APEX_EXPAND ( APEX_EVAL_SCALE_PLAN )
    APEX_EXPAND ( APEX_EVAL_SCALE_ADD_PLAN )
    APEX_EVAL_DOT_PLAN( TT1D, TT2D, TT1D )
    APEX_EVAL_DOT_PLAN( TT2D, TT2D, TT2D )
    
#undef APEX_EVAL_SCALE_ADD_PLAN
#undef APEX_EVAL_SCALE_PLAN
#undef APEX_EVAL_ADD_PLAN
       
    APEX_EXPAND ( APEX_ADD_SUPPORT_ADD_OP )           
    APEX_EXPAND2( APEX_ADD_SUPPORT_SCALE_OP )           
    // support for dot and dot.T
    APEX_ADD_SUPPORT_DOT_OP( TT2D, TT1D )
    APEX_ADD_SUPPORT_DOT_OP( TT2D, TT2D )
    
    
#undef APEX_EXPAND
#undef APEX_EXPAND2
        
};
