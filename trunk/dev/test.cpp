// test experssion template
#include <cstdio>

#ifndef _APEX_EXP_TEMPLATE_H_
#define _APEX_EXP_TEMPLATE_H_

// expression template that handles many kinds of expersssions 
// to support lazy evalution 
/*
operators supported:

Most Common:
  unary :
    A.T()                         transpose    
    A.R()                         reverse  A.R()[k][i][j] = A[k][-i][-j]        
  binary:
     A + B             elementwise add
     A - B             elementwise sub
     A * B             elementwise multiplication
     A / B             elementwise divide
     A * scalar        multiply a scalar
     dot( A, B )       matrix multiplication
     conv2( A, B, op ) op = 'V'(valid) or 'E'(equal) or 'F'(full), 2D convolution          
Less Common:
  unary:
    clone(A)                      allocate same space and clone data from A
    alloc_like(A)                 allocate a same shape of memory 
    sum_2D (A)                    sum over last 2 dimemsion 
    sigmoid(A)                    map by sigmoid function A = 1/(1+exp(A))
    sample_binary(P)              sample binary distribution with probability P
*/

namespace apex_exp_template{
    // all class names 
    template<typename Elem>
    struct Tensor;
    template<typename Elem>
    struct ReverseExp;
    template<typename Elem>
    struct TransposeExp;
    template<typename TA,typename TB>
    struct AddExp;
    template<typename TA,typename TB>
    struct SubExp;
    template<typename TA,typename TB>
    struct MulExp;
    template<typename TA,typename TB>
    struct DivExp;
    template<typename TA,typename TB>
    struct DotExp;
    template<typename TA,typename TB>
    struct Conv2Exp;
    template<typename T,typename TV>
    struct ScaleExp;
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
    namespace operators{
        // this namespace contains the operators defined
    };
};

// abstract function implementation of derived class
// the derived class must specialize the template function 
namespace apex_exp_template{
    namespace func_impl{
        // dst = s 
        template<typename T> 
        inline void fill( T &dst, double s );
    };
};
namespace apex_exp_template{            
    // basic value, and some rules for template evaluation
    template<typename Elem>
    struct Tensor{
        inline const Elem & __exp()const{
            return *static_cast<const Elem*>( this );            
        }        
        inline const TransposeExp<Elem> T() const{
            return TransposeExp<Elem>( __exp() );
        }
        inline const ReverseExp<Elem> R() const{
            return ReverseExp<Elem>( __exp() );
        }
        
        // inline implementation to make some function easier       
        template<typename TT>
        inline Elem &operator += ( const TT &exp ){
            Elem *self = static_cast<Elem*>( this );
            *self = (*self) + exp;
            return *self;
        }
        template<typename TT>
        inline Elem &operator -= ( const TT &exp ){
            Elem *self = static_cast<Elem*>( this );
            *self = (*self) - exp;
            return *self;
        }
        template<typename TT>
        inline Elem &operator *= ( const TT &exp ){
            Elem *self = static_cast<Elem*>( this );
            *self = (*self) * exp;
            return *self;
        }
        template<typename TT>
        inline Elem &operator /= ( const TT &exp ){
            Elem *self = static_cast<Elem*>( this );
            *self = (*self) / exp;
            return *self;
        }
        
        // some convention implementations
        inline Elem &__assign( double s ){
            Elem *self = static_cast<Elem*>( this );
            func_impl::fill<Elem>( *self, s );
            return self;
        }
    };
};

namespace apex_exp_template{
    // e.R : a.R[k][i][j] = a[k][-i][-j]
    template<typename Elem>
    struct ReverseExp:public Tensor< ReverseExp<Elem> >{
        const Elem &e;
        ReverseExp( const Elem &exp ):e(exp){}
        // transpose of a transpose is orignal value
        inline const Elem & R() const{
            return e;
        }
    };

    // e.T 
    template<typename Elem>
    struct TransposeExp:public Tensor< TransposeExp<Elem> >{
        const Elem &e;
        TransposeExp( const Elem &exp ):e(exp){}
        // transpose of a transpose is orignal value
        inline const Elem & T() const{
            return e;
        }
    };

    // a + b 
    template<typename TA,typename TB>
    struct AddExp:public Tensor< AddExp<TA,TB> >{
        const TA &a;
        const TB &b;
        AddExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };  
    namespace operators{
        template<typename TA,typename TB>
        inline const AddExp<TA,TB> operator+( const Tensor<TA> &a, const Tensor<TB> &b ){
            return AddExp<TA,TB>( a.__exp(), b.__exp() );
        }
    };

    // a - b 
    template<typename TA,typename TB>
    struct SubExp:public Tensor< SubExp<TA,TB> >{
        const TA &a;
        const TB &b;
        SubExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };  
    namespace operators{
        template<typename TA,typename TB>
        inline const SubExp<TA,TB> operator-( const Tensor<TA> &a, const Tensor<TB> &b ){
            return SubExp<TA,TB>( a.__exp(), b.__exp() );
        }
    };

    // a * b 
    template<typename TA,typename TB>
    struct MulExp:public Tensor< MulExp<TA,TB> >{
        const TA &a;
        const TB &b;
        MulExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };  
    namespace operators{
        template<typename TA,typename TB>
        inline const MulExp<TA,TB> operator*( const Tensor<TA> &a, const Tensor<TB> &b ){
            return MulExp<TA,TB>( a.__exp(), b.__exp() );
        }
    };

    // a / b 
    template<typename TA,typename TB>
    struct DivExp:public Tensor< DivExp<TA,TB> >{
        const TA &a;
        const TB &b;
        DivExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };
    namespace operators{
        template<typename TA,typename TB>
        inline const DivExp<TA,TB> operator/( const Tensor<TA> &a, const Tensor<TB> &b ){
            return DivExp<TA,TB>( a.__exp(), b.__exp() );
        }
    };

    // dot(a,b)
    template<typename TA,typename TB>
    struct DotExp:public Tensor< DotExp<TA,TB> >{
        const TA &a;
        const TB &b;
        DotExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };       
    namespace operators{
        template<typename TA,typename TB>
        inline const DotExp<TA,TB> dot( const Tensor<TA> &a, const Tensor<TB> &b ){
            return DotExp<TA,TB>( a.__exp(), b.__exp() );
        }
    };       

    // conv2(a,b,'V') conv2(a,b,'F') conv2(a,b,'E')
    template<typename TA,typename TB>
    struct Conv2Exp:public Tensor< Conv2Exp<TA,TB> >{
        const TA &a;
        const TB &b;
        char  op;
        Conv2Exp( const TA &ea, const TB &eb, char option ):a(ea),b(eb),op(option){}
    };       
    namespace operators{
        template<typename TA,typename TB>
        inline const Conv2Exp<TA,TB> conv2( const Tensor<TA> &a, const Tensor<TB> &b, char option ){
            return Conv2Exp<TA,TB>( a.__exp(), b.__exp(), option );
        }
    };       

    // e * s
    template<typename T,typename TV>
    struct ScaleExp:public Tensor< ScaleExp<T,TV> >{
        const T &e;
        TV s;
        ScaleExp( const T &exp, const TV &es ):e(exp),s(es){}
        inline float as_float()const{
            return (float)s;
        }
        inline double as_double()const{
            return (double)s;
        }
    };     

    namespace operators{
        // use double precision for safety concern
        template<typename T>
        inline const ScaleExp<T,double> operator*( const Tensor<T> &a, double s ){
            return ScaleExp<T,double>( a.__exp(), s );
        }
        template<typename T>
        inline const ScaleExp<T,double> operator*( double s, const Tensor<T> &a ){
            return a * s;
        }
        template<typename T>
        inline const ScaleExp<T,double> operator/( const Tensor<T> &a, double s ){
            return a * (1.0/s);
        }
        template<typename T>
        inline const ScaleExp<T,double> operator*( const ScaleExp<T,double> &a, double s ){
            return ScaleExp<T,double>( a.e, a.s * s );
        }
        template<typename T>
        inline const ScaleExp<T,double> operator*( double s, const ScaleExp<T,double> &a ){
            return a * s;
        }
        template<typename T>
        inline const ScaleExp<T,double> operator/( const ScaleExp<T,double> &a, double s ){
            return a * (1.0/s);
        }                        
    };
};
// less common operators
namespace apex_exp_template{        
    // sigmoid( e )
    template<typename Elem>
    struct SigmoidExp:public Tensor< SigmoidExp<Elem> >{
        const Elem &e;
        SigmoidExp( const Elem &exp ):e(exp){}        
    };
    namespace operators{
        template<typename T>
        inline const SigmoidExp<T> sigmoid( const Tensor<T> &e ){
            return SigmoidExp<T>( e.__exp() );
        } 
    };

    // clone( e )
    template<typename Elem>
    struct CloneExp:public Tensor< CloneExp<Elem> >{
        const Elem &e;
        CloneExp( const Elem &exp ):e(exp){}        
    };
    namespace operators{
        template<typename T>
        inline const CloneExp<T> clone( const Tensor<T> &e ){
            return CloneExp<T>( e.__exp() );
        } 
    };
    
    // alloc_like( e )
    template<typename Elem>
    struct AllocLikeExp:public Tensor< AllocLikeExp<Elem> >{
        const Elem &e;
        AllocLikeExp( const Elem &exp ):e(exp){}        
    };
    namespace operators{
        template<typename T>
        inline const AllocLikeExp<T> alloc_like( const Tensor<T> &e ){
            return AllocLikeExp<T>( e.__exp() );
        } 
    };

    // sum_2D( e )
    template<typename Elem>
    struct Sum2DExp:public Tensor< Sum2DExp<Elem> >{
        const Elem &e;
        Sum2DExp( const Elem &exp ):e(exp){}        
    };
    namespace operators{
        template<typename T>
        inline const Sum2DExp<T> sum_2D( const Tensor<T> &e ){
            return Sum2DExp<T>( e.__exp() );
        } 
    };

    // sample_binary( e )
    template<typename Elem>
    struct SampleBinaryExp:public Tensor< SampleBinaryExp<Elem> >{
        const Elem &e;
        SampleBinaryExp( const Elem &exp ):e(exp){}        
    };
    namespace operators{
        template<typename T>
        inline const SampleBinaryExp<T> sample_binary( const Tensor<T> &e ){
            return SampleBinaryExp<T>( e.__exp() );
        } 
    };

};
#endif
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
        inline void fill<test::TT>( test::TT &dst, double s ){
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

