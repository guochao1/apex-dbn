#include <cstdio>
#include "apex_exp_template.h"

struct XInt : public apex_exp_template::ContainerExp<XInt>{
    int a;
    XInt( int a ){
        this->a = a;
    }
    template<typename T>
    inline XInt& operator=( const apex_exp_template::CompositeExp<T> &exp ){
        return __assign( exp.__name_const() );
    }
    inline void __eval( const apex_exp_template::enums::AddTo &st, XInt &dst, const XInt &src ) const{
        dst.a += src.a;
    } 
};

namespace apex_exp_template{
    namespace solver_impl{
        template<typename ST>
        struct ScaleSolver<ST,XInt,double>{
            static inline void eval( XInt &dst, const XInt &src, double scalar ){
                dst.a = src.a * scalar;
            }
        };
    };
};
using namespace apex_exp_template::operators;
int main( void ){
    XInt a(1),b(2);
    a = (b*10)*10;
    printf("%d\n", a.a);
    return 0;
}
