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

#define APEX_TEMPLATE_EVAL_MAP_PLAN(T,plan_name,func_name)              \
    inline T& T::operator= ( const apex_op_plan::plan_name<T> &val ){   \
        tensor::func_name( *this, *(val.a));                            \
        return *this;                                                   \
    }                                                                   \
    
#define APEX_EVAL_SIGMOID_PLAN(T) APEX_TEMPLATE_EVAL_MAP_PLAN(T,SigmoidPlan,sigmoid) 
#define APEX_EVAL_SAMPLE_BINARY_PLAN(T) APEX_TEMPLATE_EVAL_MAP_PLAN(T,SampleBinaryPlan,sample_binary)

#define APEX_EVAL_SAMPLE_GAUSSIAN_PLAN(T)                                     \
    inline T& T::operator= ( const apex_op_plan::SampleGaussianPlan<T,TENSOR_FLOAT> &val ){ \
        tensor::sample_gaussian( *this, *(val.a), val.val );            \
        return *this;                                                   \
    }                                                                   \

#define APEX_EVAL_SCALE_ADD_PLAN(T)                                     \
    inline T& T::operator= ( const apex_op_plan::ScaleAddPlan<T,TENSOR_FLOAT> &val ){ \
        tensor::scale_add( *this, *(val.a),*(val.b), val.sa,val.sb);    \
        return *this;                                                   \
    }                                                                   \
    inline T& T::operator+= ( const apex_op_plan::ScaleAddPlan<T,TENSOR_FLOAT> &val ){ \
        tensor::sadd__scale_add( *this, *(val.a),*(val.b), val.sa,val.sb); \
        return *this;                                                   \
    }                                                                   \
    inline T& T::operator-= ( const apex_op_plan::ScaleAddPlan<T,TENSOR_FLOAT> &val ){ \
        tensor::sadd__scale_add( *this, *(val.a),*(val.b), -val.sa, -val.sb); \
        return *this;                                                   \
    }                                                                   \


#define APEX_EVAL_SCALE_PLAN(T)                                         \
    inline T& T::operator= ( const apex_op_plan::ScalePlan<T,TENSOR_FLOAT> &val ){ \
        tensor::mul( *this, *(val.a), val.scale );                      \
        return *this;                                                   \
    }                                                                   \
    inline T& T::operator+= ( const apex_op_plan::ScalePlan<T,TENSOR_FLOAT> &val ){ \
        tensor::sadd__mul( *this, *(val.a), val.scale );                \
        return *this;                                                   \
    }                                                                   \
    inline T& T::operator-= ( const apex_op_plan::ScalePlan<T,TENSOR_FLOAT> &val ){ \
        tensor::sadd__mul( *this, *(val.a), -val.scale );               \
        return *this;                                                   \
    }                                                                   \

#define APEX_EVAL_ADD_PLAN(T)                                           \
    inline T& T::operator= ( const apex_op_plan::AddPlan<T> &val ){     \
        tensor::add( *this, *(val.a), *(val.b) );                       \
        return *this;                                                   \
    }                                                                   \

#define APEX_EVAL_SUB_PLAN(T)                                           \
    inline T& T::operator= ( const apex_op_plan::SubPlan<T> &val ){     \
        tensor::sub( *this, *(val.a), *(val.b) );                       \
        return *this;                                                   \
    }                                                                   \

#define APEX_EVAL_MUL_PLAN(T)                                           \
    inline T& T::operator= ( const apex_op_plan::MulPlan<T> &val ){     \
        tensor::mul( *this, *(val.a), *(val.b) );                       \
        return *this;                                                   \
    }                                                                   \

#define APEX_EVAL_DOT_PLAN(T,TA,TB)                                     \
    inline T& T::operator= ( const apex_op_plan::DotPlan<TA,TB> &val ){ \
        tensor::dot( *this, *(val.a), *(val.b) );                       \
        return *this;                                                   \
    }                                                                   \
    inline T& T::operator+= ( const apex_op_plan::DotPlan<TA,TB> &val ){ \
        tensor::sadd__dot( *this, *(val.a), *(val.b) );                 \
        return *this;                                                   \
    }                                                                   \

#define APEX_EVAL_DOT_LT_PLAN(T,TA,TB)                                   \
    inline T& T::operator= ( const apex_op_plan::DotLTPlan<TA,TB> &val ){ \
        tensor::dot_lt( *this, *(val.a), *(val.b) );                    \
        return *this;                                                   \
    }                                                                   \
    inline T& T::operator+= ( const apex_op_plan::DotLTPlan<TA,TB> &val ){ \
        tensor::sadd__dot_lt( *this, *(val.a), *(val.b) );              \
        return *this;                                                   \
    }                                                                   \
    inline T& T::operator-= ( const apex_op_plan::DotLTPlan<TA,TB> &val ){ \
        tensor::ssub__dot_lt( *this, *(val.a), *(val.b) );              \
        return *this;                                                   \
    }                                                                   \

#define APEX_EVAL_DOT_RT_PLAN(T,TA,TB)                                   \
    inline T& T::operator= ( const apex_op_plan::DotRTPlan<TA,TB> &val ){ \
        tensor::dot_rt( *this, *(val.a), *(val.b) );                    \
        return *this;                                                   \
    }                                                                   \
    inline T& T::operator+= ( const apex_op_plan::DotRTPlan<TA,TB> &val ){ \
        tensor::sadd__dot_rt( *this, *(val.a), *(val.b) );              \
        return *this;                                                   \
    }                                                                   \
    
    APEX_EXPAND(  APEX_EVAL_SIGMOID_PLAN )
    APEX_EXPAND(  APEX_EVAL_SAMPLE_BINARY_PLAN )
    APEX_EXPAND(  APEX_EVAL_SAMPLE_GAUSSIAN_PLAN )
    APEX_EXPAND(  APEX_EVAL_ADD_PLAN )
    APEX_EXPAND(  APEX_EVAL_SUB_PLAN )
    APEX_EXPAND(  APEX_EVAL_MUL_PLAN )
    APEX_EXPAND ( APEX_EVAL_SCALE_PLAN )
    APEX_EXPAND ( APEX_EVAL_SCALE_ADD_PLAN )
    APEX_EVAL_DOT_PLAN( TT1D, TT1D, TT2D )
    APEX_EVAL_DOT_PLAN( TT2D, TT2D, TT2D )
    APEX_EVAL_DOT_LT_PLAN( TT2D, TT1D, TT1D )
    APEX_EVAL_DOT_RT_PLAN( TT1D, TT1D, TT2D )
    APEX_EVAL_DOT_RT_PLAN( TT2D, TT2D, TT2D )
    
    
#define APEX_EVAL_CLONE_PLAN_1D(T,plan,op)                              \
    inline TT1D& TT1D::operator= ( const apex_op_plan::plan<T> &val ){  \
        this->set_param( val.a->x_max );                                \
        tensor::alloc_space( *this );                                   \
        op;                                                             \
        return *this;                                                   \
    }                                                                   \

#define APEX_EVAL_CLONE_PLAN_2D(T,plan,op)                              \
    inline TT2D& TT2D::operator= ( const apex_op_plan::plan<T> &val ){  \
        this->set_param( val.a->y_max,val.a->x_max );                   \
        tensor::alloc_space( *this );                                   \
        op;                                                             \
        return *this;                                                   \
    }                                                                   \

#define APEX_EVAL_CLONE_PLAN_3D(T,plan,op)                              \
    inline TT3D& TT3D::operator= ( const apex_op_plan::plan<T> &val ){  \
        this->set_param( val.a->z_max, val.a->y_max,val.a->x_max );     \
        tensor::alloc_space( *this );                                   \
        op;                                                             \
        return *this;                                                   \
    }                                                                   \

#define APEX_EVAL_CLONE_PLAN_4D(T,plan,op)                              \
    inline TT4D& TT4D::operator= ( const apex_op_plan::plan<T> &val ){  \
        this->set_param( val.a->h_max,val.a->z_max,val.a->y_max,val.a->x_max ); \
        tensor::alloc_space( *this );                                   \
        op;                                                             \
        return *this;                                                   \
    }                                                                   \
    
    APEX_EVAL_CLONE_PLAN_1D(CTensor1D,ClonePlan,tensor::copy(*this,*(val.a)));
    APEX_EVAL_CLONE_PLAN_1D(CTensor1D,AllocLikePlan, );
    APEX_EVAL_CLONE_PLAN_1D(GTensor1D,AllocLikePlan, );
    APEX_EVAL_CLONE_PLAN_2D(CTensor2D,ClonePlan,tensor::copy(*this,*(val.a)));
    APEX_EVAL_CLONE_PLAN_2D(CTensor2D,AllocLikePlan,);
    APEX_EVAL_CLONE_PLAN_2D(GTensor2D,AllocLikePlan,);
    APEX_EVAL_CLONE_PLAN_3D(CTensor3D,ClonePlan,tensor::copy(*this,*(val.a)));
    APEX_EVAL_CLONE_PLAN_3D(CTensor3D,AllocLikePlan,);
    APEX_EVAL_CLONE_PLAN_3D(GTensor3D,AllocLikePlan,);
    APEX_EVAL_CLONE_PLAN_4D(CTensor4D,ClonePlan,tensor::copy(*this,*(val.a)));
    APEX_EVAL_CLONE_PLAN_4D(CTensor4D,AllocLikePlan,);
    APEX_EVAL_CLONE_PLAN_4D(GTensor4D,AllocLikePlan,);
    

#undef APEX_EVAL_SCALE_ADD_PLAN
#undef APEX_EVAL_SCALE_PLAN
#undef APEX_EVAL_ADD_PLAN
       
    APEX_EXPAND ( APEX_ADD_SUPPORT_SIGMOID_OP )           
    APEX_EXPAND ( APEX_ADD_SUPPORT_CLONE_OP )           
    APEX_EXPAND ( APEX_ADD_SUPPORT_ALLOC_LIKE_OP )           
    APEX_EXPAND ( APEX_ADD_SUPPORT_SAMPLE_BINARY_OP )           
    APEX_EXPAND ( APEX_ADD_SUPPORT_ADD_OP )              
    APEX_EXPAND ( APEX_ADD_SUPPORT_SUB_OP )              
    APEX_EXPAND ( APEX_ADD_SUPPORT_MUL_OP )              
    APEX_EXPAND ( APEX_ADD_SUPPORT_TRANSPOSE_OP )
    APEX_EXPAND2( APEX_ADD_SUPPORT_SCALE_OP )           
    APEX_EXPAND2( APEX_ADD_SUPPORT_SAMPLE_GAUSSIAN_OP )           

    // support for dot and dot.T
    APEX_ADD_SUPPORT_DOT_OP   ( TT1D, TT2D )
    APEX_ADD_SUPPORT_DOT_OP   ( TT2D, TT2D )    
    APEX_ADD_SUPPORT_DOT_LT_OP( TT1D, TT1D ) 
    APEX_ADD_SUPPORT_DOT_RT_OP( TT1D, TT2D ) 
    APEX_ADD_SUPPORT_DOT_RT_OP( TT2D, TT2D ) 


    
#undef APEX_EXPAND
#undef APEX_EXPAND2
    
    
    APEX_ADD_SUPPORT_SUM_2D_OP( TT3D )    

    inline TT1D& TT1D::operator+= ( const apex_op_plan::Sum2DPlan<TT3D> &val ){  
        tensor::crbm::sadd__sum_2D( *this, *(val.a) );
        return *this;                                                   
    }                                                                   
    inline TT1D& TT1D::operator-= ( const apex_op_plan::Sum2DPlan<TT3D> &val ){  
        tensor::crbm::ssub__sum_2D( *this, *(val.a) );
        return *this;                                                   
    }                                                                   
};


namespace apex_tensor{
    inline TSIDX2D& TSIDX2D::operator=( const apex_op_plan::ClonePlan<CSparseIndex2D> &val ){
        this->length = (val.a)->length;
        this->alloc_length = (val.a)->alloc_length;
        tensor::alloc_space_index( *this );
        tensor::copy_index( *this, *(val.a) );
        return *this;
    } 
};

// support for sparse operation
namespace apex_tensor{    
    APEX_ADD_SUPPORT_CLONE_OP    ( TSIDX2D )
    APEX_ADD_SUPPORT_TRANSPOSE_OP( TT2DS )
    APEX_ADD_SUPPORT_TRANSPOSE_OP( TT1DS )
    APEX_ADD_SUPPORT_DOT_LT_OP   ( TT2DS, TT2D )
    APEX_ADD_SUPPORT_DOT_OP      ( TT2DS, TT2D ) 
    APEX_ADD_SUPPORT_DOT_LT_OP   ( TT1DS, TT1D )
    APEX_ADD_SUPPORT_DOT_OP      ( TT1DS, TT2D )
    APEX_ADD_SUPPORT_SCALE_OP    ( TT1DS, TENSOR_FLOAT )
    APEX_ADD_SUPPORT_SUB_OP      ( TT2DS )
    APEX_EVAL_SUB_PLAN           ( TT2DS )    

    APEX_ADD_SUPPORT_SCALE_DOT_LT_OP ( TT1DS, TT1D, TENSOR_FLOAT )
};

namespace apex_tensor{
    inline TT1D& TT1D::operator+= ( const apex_op_plan::ScalePlan<TT1DS,TENSOR_FLOAT> &val ){
        tensor::sadd__mul( *this, *(val.a), val.scale );                      
        return *this;                                                   
    }                                                                       

    inline TT2D& TT2D::operator+= ( const apex_op_plan::ScalePlan<apex_op_plan::DotLTPlan<TT1DS,TT1D>,TENSOR_FLOAT> &val ){
        tensor::sadd__dot_lt_scale( *this, *((val.a)->a), *((val.a)->b),  val.scale );                      
        return *this;                                                   
    }                                                                       
};
namespace apex_tensor{
    APEX_EVAL_DOT_PLAN   ( TT1D, TT1DS, TT2D  )
    APEX_EVAL_DOT_PLAN   ( TT2D, TT2DS, TT2D  )
    APEX_EVAL_DOT_RT_PLAN( TT2DS, TT2D, TT2D  )
    APEX_EVAL_DOT_LT_PLAN( TT2D , TT1DS, TT1D )
    APEX_EVAL_DOT_LT_PLAN( TT2D , TT2DS, TT2D )    
};

