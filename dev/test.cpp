// test experssion template
#include "apex_exp_template.h"

#include <cstdio>
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
    };
};

using namespace apex_exp_template::operators;

int main( void ){
    test::TT a(1),b(200),c(3);
    //c = b;
    //c = conv2( a, b.R(), 'V' );
    printf("c=%f\n", c.a );
    return 0;    
}

