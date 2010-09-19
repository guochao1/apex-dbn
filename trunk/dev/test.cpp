#include <cstdio>
#include "apex_exp_template.h"

/*! \brief test class of expression template, print the operations */
struct XVar : public apex_exp_template::ContainerExp<XVar>{
    const char *name;
    XVar( const char *name ){
        this->name = name;
    }
    template<typename T>
    inline XVar& operator=( const apex_exp_template::CompositeExp<T> &exp ){
        return __assign( exp.__name_const() );
    }
    inline void __eval( const apex_exp_template::enums::AddTo &st, XVar &dst, const XVar &src ) const{
        
    } 
};

namespace apex_exp_template{
    namespace solver_impl{
        template<typename ST,typename OP>
        struct ScalarMapSolver<ST,OP,XVar,double>{
            static inline void eval( XVar &dst, const XVar &src, double scalar ){
                printf("%s %s %s %s %lf\n", dst.name, ST::str, src.name, OP::str, scalar );
            }
        };

        template<typename ST,typename OP>
        struct BinaryMapSolver<ST,OP,XVar,XVar,XVar>{
            static inline void eval( XVar &dst, const XVar &a, const XVar &b ){
                printf("%s %s %s %s %s\n", dst.name, ST::str, a.name, OP::str, b.name );
            }
        };
        
        template<typename ST,bool ta, bool tb>
        struct DotSolver<ST,XVar, XVar,XVar,ta,tb>{
            static inline void eval( XVar &dst, const XVar &a, const XVar &b ){
                printf("%s %s dot( %s%s, %s%s )\n", dst.name,ST::str, a.name, ta?".T":"" , b.name, tb?".T":"" );
            }
        };

        template<typename ST,bool ta, bool tb>
        struct Conv2Solver<ST,XVar, XVar,XVar,ta,tb>{
            static inline void eval( XVar &dst, const XVar &a, const XVar &b, char option ){
                printf("%s %s conv2( %s%s, %s%s, '%c' )\n", dst.name,ST::str, a.name, ta?".R":"" , b.name, tb?".R":"", option );
            }
        };
    };
};

using namespace apex_exp_template::operators;
int main( void ){
    XVar a("a"),b("b");
    a = (a+b)-3;
    a = dot(a, b.T())*30;
    a += dot(a.T(), b.T());
    a = dot(a.T(), b); 
    a = conv2( a.R(), b, 'V' );
    return 0;
}
