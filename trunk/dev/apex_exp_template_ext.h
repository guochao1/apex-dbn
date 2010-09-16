#ifndef _APEX_EXP_TEMPLATE_EXT_H_
#define _APEX_EXP_TEMPLATE_EXT_H_

#include "apex_exp_template.h"

// extension of basic expression template
namespace apex_exp_template{
    // less common
    template<typename Elem>
    struct CloneExp;
    template<typename Elem>
    struct AllocLikeExp;
    template<typename Elem>
    struct Sum2DExp;
    template<typename Elem>
    struct SigmoidExp;
    template<typename Elem>
    struct SampleBinaryExp;    
};

// less common operators
namespace apex_exp_template{        
    // sigmoid( e )
    template<typename Elem>
    struct SigmoidExp:public Exp< SigmoidExp<Elem> >{
        const Elem &e;
        SigmoidExp( const Elem &exp ):e(exp){}        
    };
    namespace operators{
        template<typename T>
        inline const SigmoidExp<T> sigmoid( const Exp<T> &e ){
            return SigmoidExp<T>( e.__exp() );
        } 
    };

    // clone( e )
    template<typename Elem>
    struct CloneExp:public Exp< CloneExp<Elem> >{
        const Elem &e;
        CloneExp( const Elem &exp ):e(exp){}        
    };
    namespace operators{
        template<typename T>
        inline const CloneExp<T> clone( const Exp<T> &e ){
            return CloneExp<T>( e.__exp() );
        } 
    };
    
    // alloc_like( e )
    template<typename Elem>
    struct AllocLikeExp:public Exp< AllocLikeExp<Elem> >{
        const Elem &e;
        AllocLikeExp( const Elem &exp ):e(exp){}        
    };
    namespace operators{
        template<typename T>
        inline const AllocLikeExp<T> alloc_like( const Exp<T> &e ){
            return AllocLikeExp<T>( e.__exp() );
        } 
    };

    // sum_2D( e )
    template<typename Elem>
    struct Sum2DExp:public Exp< Sum2DExp<Elem> >{
        const Elem &e;
        Sum2DExp( const Elem &exp ):e(exp){}        
    };
    namespace operators{
        template<typename T>
        inline const Sum2DExp<T> sum_2D( const Exp<T> &e ){
            return Sum2DExp<T>( e.__exp() );
        } 
    };

    // sample_binary( e )
    template<typename Elem>
    struct SampleBinaryExp:public Exp< SampleBinaryExp<Elem> >{
        const Elem &e;
        SampleBinaryExp( const Elem &exp ):e(exp){}        
    };
    namespace operators{
        template<typename T>
        inline const SampleBinaryExp<T> sample_binary( const Exp<T> &e ){
            return SampleBinaryExp<T>( e.__exp() );
        } 
    };
};
#endif

