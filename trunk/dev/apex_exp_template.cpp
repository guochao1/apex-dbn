#include <cstdio>
#include "apex_exp_template.h"

/*! \file apex_exp_template.cpp
 *  \brief a showcase for how to use expression template
 *  print out all the operations
 */

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

        template<typename ST,bool ta, bool tb, typename CT>
        struct Conv2Solver<ST,XVar, XVar,XVar,ta,tb,CT>{
            static inline void eval( XVar &dst, const XVar &a, const XVar &b ){
                printf("%s %s conv2( %s%s, %s%s, '%s' )\n", dst.name,ST::str, a.name, ta?".R":"" , b.name, tb?".R":"", CT::str );
            }
        };
    };
};

using namespace apex_exp_template::enums;
using namespace apex_exp_template::operators;
int main( void ){
    XVar a("a"),b("b");
    a /= conv2( a.R(), b, Valid::op );
    a = ((a - b)/3-2+1)*10;
    a = dot( a, b.T())*3;    
    return 0;
}
