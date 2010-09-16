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
    struct Exp;
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
    namespace operators{
        // this namespace contains the operators defined
    };
};

// abstract function implementation of derived class
// the derived class must specialize the template function 
namespace apex_exp_template{
    namespace store_method{
        const int SAVE = 0;
        const int ADD  = 1;
        const int SUB  = 3;
        const int MUL  = 5;
    };
    namespace func_impl{
        // dst = s 
        template<typename T> 
        inline void scalar( int store_op, T &dst, double s );
    };
};

namespace apex_exp_template{            
    // basic value, and some rules for template evaluation
    template<typename Elem>
    struct Exp{
        inline const Elem & __exp()const{
            return *static_cast<const Elem*>( this );            
        }        
        inline const TransposeExp<Elem> T() const{
            return TransposeExp<Elem>( __exp() );
        }
        inline const ReverseExp<Elem> R() const{
            return ReverseExp<Elem>( __exp() );
        }        
    };  
    template<typename Elem>
    
    // container that can be assigned to values
    template<typename Elem>
    struct ContainerExp: public Exp<Elem>{
        // inline implementation to make some function easier       
        template<typename TT>
        inline Elem &operator += ( const TT &exp ){
            return __assign( store_method::ADD, exp );
        }
        template<typename TT>
        inline Elem &operator -= ( const TT &exp ){
            return __assign( store_method::SUB, exp );
        }
        template<typename TT>
        inline Elem &operator *= ( const TT &exp ){
            return __assign( store_method::MUL, exp );
        }
        
        // some convention implementations
        inline Elem &__assign( int st_op, double &s ){
            Elem *self = static_cast<Elem*>( this );
            func_impl::scalar<Elem>( st_op, *self, s );
            return self;
        }
    };
};

namespace apex_exp_template{
    // e.R : a.R[k][i][j] = a[k][-i][-j]
    template<typename Elem>
    struct ReverseExp:public Exp< ReverseExp<Elem> >{
        const Elem &e;
        ReverseExp( const Elem &exp ):e(exp){}
        // transpose of a transpose is orignal value
        inline const Elem & R() const{
            return e;
        }
    };

    // e.T 
    template<typename Elem>
    struct TransposeExp:public Exp< TransposeExp<Elem> >{
        const Elem &e;
        TransposeExp( const Elem &exp ):e(exp){}
        // transpose of a transpose is orignal value
        inline const Elem & T() const{
            return e;
        }
    };

    // a + b 
    template<typename TA,typename TB>
    struct AddExp:public Exp< AddExp<TA,TB> >{
        const TA &a;
        const TB &b;
        AddExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };  
    namespace operators{
        template<typename TA,typename TB>
        inline const AddExp<TA,TB> operator+( const Exp<TA> &a, const Exp<TB> &b ){
            return AddExp<TA,TB>( a.__exp(), b.__exp() );
        }
    };

    // a - b 
    template<typename TA,typename TB>
    struct SubExp:public Exp< SubExp<TA,TB> >{
        const TA &a;
        const TB &b;
        SubExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };  
    namespace operators{
        template<typename TA,typename TB>
        inline const SubExp<TA,TB> operator-( const Exp<TA> &a, const Exp<TB> &b ){
            return SubExp<TA,TB>( a.__exp(), b.__exp() );
        }
    };

    // a * b 
    template<typename TA,typename TB>
    struct MulExp:public Exp< MulExp<TA,TB> >{
        const TA &a;
        const TB &b;
        MulExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };  
    namespace operators{
        template<typename TA,typename TB>
        inline const MulExp<TA,TB> operator*( const Exp<TA> &a, const Exp<TB> &b ){
            return MulExp<TA,TB>( a.__exp(), b.__exp() );
        }
    };

    // a / b 
    template<typename TA,typename TB>
    struct DivExp:public Exp< DivExp<TA,TB> >{
        const TA &a;
        const TB &b;
        DivExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };
    namespace operators{
        template<typename TA,typename TB>
        inline const DivExp<TA,TB> operator/( const Exp<TA> &a, const Exp<TB> &b ){
            return DivExp<TA,TB>( a.__exp(), b.__exp() );
        }
    };

    // dot(a,b)
    template<typename TA,typename TB>
    struct DotExp:public Exp< DotExp<TA,TB> >{
        const TA &a;
        const TB &b;
        DotExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };       
    namespace operators{
        template<typename TA,typename TB>
        inline const DotExp<TA,TB> dot( const Exp<TA> &a, const Exp<TB> &b ){
            return DotExp<TA,TB>( a.__exp(), b.__exp() );
        }
    };       

    // conv2(a,b,'V') conv2(a,b,'F') conv2(a,b,'E')
    template<typename TA,typename TB>
    struct Conv2Exp:public Exp< Conv2Exp<TA,TB> >{
        const TA &a;
        const TB &b;
        char  op;
        Conv2Exp( const TA &ea, const TB &eb, char option ):a(ea),b(eb),op(option){}
    };       
    namespace operators{
        template<typename TA,typename TB>
        inline const Conv2Exp<TA,TB> conv2( const Exp<TA> &a, const Exp<TB> &b, char option ){
            return Conv2Exp<TA,TB>( a.__exp(), b.__exp(), option );
        }
    };       

    // e * s
    template<typename T,typename TV>
    struct ScaleExp:public Exp< ScaleExp<T,TV> >{
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
        inline const ScaleExp<T,double> operator*( const Exp<T> &a, double s ){
            return ScaleExp<T,double>( a.__exp(), s );
        }
        template<typename T>
        inline const ScaleExp<T,double> operator*( double s, const Exp<T> &a ){
            return a * s;
        }
        template<typename T>
        inline const ScaleExp<T,double> operator/( const Exp<T> &a, double s ){
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
#endif

