#ifndef _APEX_OP_PLAN_H_
#define _APEX_OP_PLAN_H_

// plans to extend more operators
namespace apex_op_plan{

    template<typename T>
    struct AddPlan{
        const T *a, *b;
        AddPlan( const T *a, const T *b ) {
            this->a = a;
            this->b = b;
        } 
    };
    template<typename T,typename TV>
    struct ScalePlan{
        const T *a;
        TV scale;
        ScalePlan( const T *a, TV scale ) {
            this->a = a;
            this->scale = scale;
        } 
    };

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

