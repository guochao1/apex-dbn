// test experssion template
#include "apex_exp_template.h"

#include <cstdio>
namespace test{
    namespace _at = apex_exp_template;       

    struct TT:public _at::Tensor<TT>{
        float a;
        TT( float a ){
            this->a = a;
        }   
        
        inline TT& __assign( const TT &a ){
            this->a = a.a;
        }

        inline TT&operator=( const _at::Tensor<TT> &a ){
            return __assign( a.__exp() );
        }

        inline TT&operator=( const _at::TransposeExp<TT> &e ){
            a = e.e.a;
            return *this;
        }
        inline TT&operator=( const _at::MulExp<TT,TT > &e ){
            a = e.a.a+e.b.a;
            return *this;
        }
        inline TT&operator=( const _at::ScaleExp<TT,double > &e ){
            a = e.e.a * e.s;
            return *this;
        }
        inline TT&operator=( const _at::SampleBinaryExp<TT> &e ){
            a = e.e.a*10;
        }
    };    
    using namespace apex_exp_template::operators;
};

namespace apex_exp_template{
    namespace func_impl{
        template<>
        inline void scalar<test::TT>( int st_op, test::TT &dst, double s ){
            dst.a = s;
        } 
    };
};

using namespace apex_exp_template::operators;

int main( void ){
    test::TT a(1),b(200),c(3);
    c = b;
    //c = conv2( a, b.R(), 'V' );
    printf("c=%f\n", c.a );
    return 0;    
}

