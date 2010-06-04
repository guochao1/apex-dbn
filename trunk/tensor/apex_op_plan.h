#ifndef _APEX_OP_PLAN_H_
#define _APEX_OP_PLAN_H_

// plans to extend more operators
namespace apex_op_plan{

#define APEX_TEMPLATE_MAP_PLAN(plan_name)       \
    struct plan_name{                           \
        const T *a;                             \
        plan_name( const T *a ){                \
            this->a = a;                        \
        }                                       \
    };                                          \

#define APEX_TEMPLATE_ADD_SUPPORT_OP(T,plan_name,map_name)      \
    inline apex_op_plan::plan_name<T> map_name( const T & a ){  \
        return apex_op_plan::plan_name<T>( &a );                \
    }                                                           \

    template<typename T>
    APEX_TEMPLATE_MAP_PLAN( SigmoidPlan )
#define APEX_ADD_SUPPORT_SIGMOID_OP(T) APEX_TEMPLATE_ADD_SUPPORT_OP(T,SigmoidPlan,sigmoid)  
    template<typename T>
    APEX_TEMPLATE_MAP_PLAN( SampleBinaryPlan )
#define APEX_ADD_SUPPORT_SAMPLE_BINARY_OP(T) APEX_TEMPLATE_ADD_SUPPORT_OP(T,SampleBinaryPlan,sample_binary)  
    template<typename T>
    APEX_TEMPLATE_MAP_PLAN( ClonePlan )
#define APEX_ADD_SUPPORT_CLONE_OP(T) APEX_TEMPLATE_ADD_SUPPORT_OP(T,ClonePlan,clone)  
    template<typename T>
    APEX_TEMPLATE_MAP_PLAN( AllocLikePlan )
#define APEX_ADD_SUPPORT_ALLOC_LIKE_OP(T) APEX_TEMPLATE_ADD_SUPPORT_OP(T,AllocLikePlan,alloc_like)  
    template<typename T>
    APEX_TEMPLATE_MAP_PLAN( Sum2DPlan )
#define APEX_ADD_SUPPORT_SUM_2D_OP(T) APEX_TEMPLATE_ADD_SUPPORT_OP(T,Sum2DPlan,sum_2D)  

};

namespace apex_op_plan{

#define APEX_TEMPLATE_MAP_PLAN_B(plan_name)     \
    struct plan_name{                           \
        const T *a;                             \
        TV val;                                 \
        plan_name( const T *a, TV val ){        \
            this->a = a; this->val = val;       \
        }                                       \
    };                                          \

#define APEX_TEMPLATE_ADD_SUPPORT_OP_B(T,TV,plan_name,map_name)         \
    inline apex_op_plan::plan_name<T,TV> map_name( const T & a, TV b ){ \
        return apex_op_plan::plan_name<T,TV>( &a, b );                  \
    }                                                                   \

    template<typename T,typename TV>
    APEX_TEMPLATE_MAP_PLAN_B( SampleGaussianPlan )
#define APEX_ADD_SUPPORT_SAMPLE_GAUSSIAN_OP(T,TV) APEX_TEMPLATE_ADD_SUPPORT_OP_B(T,TV,SampleGaussianPlan,sample_gaussian)  

};

namespace apex_op_plan{
    template<typename T>
    struct AddPlan{
        const T *a, *b;
        AddPlan( const T *a, const T *b ) {
            this->a = a;
            this->b = b;
        } 
    };
    
#define APEX_ADD_SUPPORT_ADD_OP(T)                                      \
    inline apex_op_plan::AddPlan<T> operator+( const T &a, const T &b ){ \
        return apex_op_plan::AddPlan<T>( &a, &b );                      \
    }                                                                   \

    template<typename T>
    struct SubPlan{
        const T *a, *b;
        SubPlan( const T *a, const T *b ) {
            this->a = a;
            this->b = b;
        } 
    };
    
#define APEX_ADD_SUPPORT_SUB_OP(T)                                      \
    inline apex_op_plan::SubPlan<T> operator-( const T &a, const T &b ){ \
        return apex_op_plan::SubPlan<T>( &a, &b );                      \
    }                                                                   \

    template<typename T>
    struct MulPlan{
        const T *a, *b;
        MulPlan( const T *a, const T *b ) {
            this->a = a;
            this->b = b;
        } 
    };
    
#define APEX_ADD_SUPPORT_MUL_OP(T)                                      \
    inline apex_op_plan::MulPlan<T> operator*( const T &a, const T &b ){ \
        return apex_op_plan::MulPlan<T>( &a, &b );                      \
    }                                                                   \

    template<typename T,typename TV>
    struct ScalePlan{
        const T *a;
        TV scale;
        ScalePlan( const T *a, TV scale ) {
            this->a = a;
            this->scale = scale;
        } 
    };

#define APEX_ADD_SUPPORT_SCALE_OP(T,TV)                                 \
    inline apex_op_plan::ScalePlan<T,TV> operator*( const T &a, TV scale ){ \
        return apex_op_plan::ScalePlan<T,TV>( &a, scale );              \
    }                                                                   \
    inline apex_op_plan::ScalePlan<T,TV> operator*( TV scale, const T &a ){ \
        return apex_op_plan::ScalePlan<T,TV>( &a, scale );              \
    }                                                                   \

#define APEX_ADD_SUPPORT_SCALE_DOT_OP(TA,TB,TV)                         \
    inline apex_op_plan::ScalePlan<apex_op_plan::DotPlan<TA,TB> ,TV> operator*( const apex_op_plan::DotPlan<TA,TB> &a, TV scale ){ \
        return apex_op_plan::ScalePlan<apex_op_plan::DotPlan<TA,TB> ,TV>( &a, scale ); \
    }                                                                   \
    inline apex_op_plan::ScalePlan<apex_op_plan::DotPlan<TA,TB> ,TV> operator*( TV scale, const apex_op_plan::DotPlan<TA,TB> &a ){ \
        return apex_op_plan::ScalePlan<apex_op_plan::DotPlan<TA,TB> ,TV>( &a, scale ); \
    }                                                                   \
        
#define APEX_ADD_SUPPORT_SCALE_DOT_LT_OP(TA,TB,TV)                      \
    inline apex_op_plan::ScalePlan<apex_op_plan::DotLTPlan<TA,TB> ,TV> operator*( const apex_op_plan::DotLTPlan<TA,TB> &a, TV scale ){ \
        return apex_op_plan::ScalePlan<apex_op_plan::DotLTPlan<TA,TB> ,TV>( &a, scale ); \
    }                                                                   \
    inline apex_op_plan::ScalePlan<apex_op_plan::DotLTPlan<TA,TB> ,TV> operator*( TV scale, const apex_op_plan::DotLTPlan<TA,TB> &a ){ \
        return apex_op_plan::ScalePlan<apex_op_plan::DotLTPlan<TA,TB> ,TV>( &a, scale ); \
    }                                                                   \



    template<typename TA,typename TB>
    struct DotPlan{
        const TA *a;
        const TB *b;
        DotPlan( const TA *a, const TB *b ) {
            this->a = a;
            this->b = b;
        } 
    };

#define APEX_ADD_SUPPORT_DOT_OP(TA,TB)                                  \
    inline apex_op_plan::DotPlan<TA,TB> dot( const TA &a, const TB &b ){\
        return apex_op_plan::DotPlan<TA,TB>( &a,&b );                   \
    }                                                                   \

    template<typename T>
    struct TransposePlan{
        const T *mat;
        TransposePlan( const T *mat ){
            this->mat = mat;
        }
    };    

#define APEX_ADD_SUPPORT_TRANSPOSE_OP(TP)                               \
    inline apex_op_plan::TransposePlan<TP> TP::T()const{                \
        return apex_op_plan::TransposePlan<TP>( this );                 \
    }                                                                   \
    
    template<typename TA,typename TB>
    struct DotLTPlan{
        const TA *a;
        const TB *b;
        DotLTPlan( const TA *a, const TB *b ) {
            this->a = a;
            this->b = b;
        } 
    };

#define APEX_ADD_SUPPORT_DOT_LT_OP(TA,TB)                               \
    inline apex_op_plan::DotLTPlan<TA,TB> dot( const apex_op_plan::TransposePlan<TA> &a, const TB &b ){ \
        return apex_op_plan::DotLTPlan<TA,TB>( a.mat, &b );              \
    }                                                                   \

    template<typename TA,typename TB>
    struct DotRTPlan{
        const TA *a;
        const TB *b;
        DotRTPlan( const TA *a, const TB *b ) {
            this->a = a;
            this->b = b;
        } 
    };

#define APEX_ADD_SUPPORT_DOT_RT_OP(TA,TB)                               \
    inline apex_op_plan::DotRTPlan<TA,TB> dot( const TA &a, const apex_op_plan::TransposePlan<TB> &b ){ \
        return apex_op_plan::DotRTPlan<TA,TB>( &a, b.mat );              \
    }                                                                   \

           
    template<typename T,typename TV>
    struct ScaleAddPlan{
        const T *a,*b;
        TV       sa,sb;
        ScaleAddPlan( const T *a, const T *b, TV sa, TV sb ) {
            this->a = a;
            this->b = b;
            this->sa = sa;
            this->sb = sb;
        } 
    };
    template<typename T,typename TV>
    inline ScaleAddPlan<T,TV> operator+( const ScalePlan<T,TV> &aa, const ScalePlan<T,TV> &bb ){
        return ScaleAddPlan<T,TV>( aa.a, bb.a, aa.scale, bb.scale );
    }
        
};

#endif

