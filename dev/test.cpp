// test experssion template


#include <cstdio>
#include "apex_exp_template.h"
namespace test{

    struct TT:public apex_exp_template::ContainerExp<TT>{
        float a;
        TT( float a ){
            this->a = a;
        }   
        inline TT& operator=( double s ){
            return __assign( s );
        }
        template<typename Elem>
        inline TT& operator=( const apex_exp_template::CompositeExp<Elem> &a ){
            return __assign( a.__name() );
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
        template<>
        inline void unary<test::TT, test::TT>( int st_op, test::TT &dst, const test::TT &src ){
            dst.a += src.a;
        }
        template<>
        inline void scale<test::TT, test::TT>( int st_op, test::TT &dst, const test::TT &src, double s ){
            dst.a += src.a*s;
        }
        template<>
        inline void add<test::TT, test::TT, test::TT>( int st_op, test::TT &dst, const test::TT &a, const test::TT &b ){
            dst.a = a.a+b.a;
        }
        
    };
};

using namespace apex_exp_template::operators;

int main( void ){
    test::TT a(1),b(200),c(3);
    c *= (a+b)*2.0;
    //c = b;
    //c = conv2( a, b.R(), 'V' );
    printf("c=%f\n", c.a );
    return 0;    
}

