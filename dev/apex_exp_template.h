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
    template<typename Name,typename Alias>
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
        const int DIV  = 7;        
    };
    namespace func_impl{
        // unary operators
        template<typename T> 
        inline void scalar( int store_op, T &dst, double s );
        template<typename T,typename TA>
        inline void unary ( int store_op, T &dst, const TA &src );
        template<typename T,typename TA>
        inline void scale ( int store_op, T &dst, const TA &src, double s );
        // binary operators
        template<typename T,typename TA,typename TB>
        inline void add   ( int store_op, T &dst, const TA &a, const TB &b );
        template<typename T,typename TA,typename TB>
        inline void sub   ( int store_op, T &dst, const TA &a, const TB &b );
        template<typename T,typename TA,typename TB>
        inline void mul   ( int store_op, T &dst, const TA &a, const TB &b );
        template<typename T,typename TA,typename TB>
        inline void div   ( int store_op, T &dst, const TA &a, const TB &b );
        template<typename T,typename TA,typename TB>
        inline void dot   ( int store_op, T &dst, const TA &a, const TB &b, bool transposeA, bool transposeB );
        template<typename T,typename TA,typename TB>
        inline void conv2 ( int store_op, T &dst, const TA &a, const TB &b, bool reverseA, bool reverseB, char option );        
    };
};

namespace apex_exp_template{            
    // basic value, and some rules for template evaluation
    template<typename Name, typename Alias>
    struct Exp{
        inline const Name &  __name()const{
            return *static_cast<const Name*>( this );            
        }
        inline const Alias & __alias()const{
            return *static_cast<const Alias*>( this );            
        }
        inline const TransposeExp<Alias> T()const{
            return TransposeExp<Alias>( __alias() );
        }
        inline const ReverseExp<Alias> R()const{
            return ReverseExp<Alias>( __alias() );
        }        
    };  
    
    template<typename Elem>
    struct CompositeExp: public Exp< Elem, CompositeExp<Elem> >{
    };

    // container that can be assigned to values
    template<typename Elem>
    struct ContainerExp: public Exp< Elem, ContainerExp<Elem> >{
    private:
        inline Elem & __self(){
            return *static_cast<Elem*>( this );            
        }
        // inline implementation to make some function easier         
    public:
        // d = scalar
        inline Elem & __assign( double s ){
            func_impl::scalar<Elem>( store_method::SAVE, __self(), s );
            return __self();
        }
        inline Elem & operator+=( double s ){
            func_impl::scalar<Elem>( store_method::ADD, __self(), s );
            return __self();
        }
        inline Elem & operator-=( double s ){
            func_impl::scalar<Elem>( store_method::ADD, __self(), -s );
            return __self();
        }
        inline Elem & operator*=( double s ){
            func_impl::scalar<Elem>( store_method::MUL, __self(), s );
            return __self();
        }
        inline Elem & operator/=( double s ){
            func_impl::scalar<Elem>( store_method::MUL, __self(), 1.0/s );
            return __self();
        }
        // d = src
        template<typename T>
        inline Elem & operator+=( const ContainerExp<T> &e ){
            func_impl::unary<Elem,T>( store_method::ADD, __self(), e.__name() );
            return __self();
        }
        template<typename T>
        inline Elem & operator-=( const ContainerExp<T> &e ){
            func_impl::unary<Elem,T>( store_method::SUB, __self(), e.__name() );
            return __self();
        }
        template<typename T>
        inline Elem & operator*=( const ContainerExp<T> &e ){
            func_impl::unary<Elem,T>( store_method::MUL, __self(), e.__name() );
            return __self();
        }
        template<typename T>
        inline Elem & operator/=( const ContainerExp<T> &e ){
            func_impl::unary<Elem,T>( store_method::DIV, __self(), e.__name() );
            return __self();
        }        
    public:
        // push every thing to function assign
        template<typename T>
        inline Elem & __assign( const CompositeExp<T> & e){
            return __assign( store_method::SAVE, e.__name() );
        }
        template<typename T>
        inline Elem & operator+=( const CompositeExp<T> & e){
            return __assign( store_method::ADD, e.__name() );
        }
        template<typename T>
        inline Elem & operator-=( const CompositeExp<T> & e){
            return __assign( store_method::SUB, e.__name() );
        }
        template<typename T>
        inline Elem & operator*=( const CompositeExp<T> & e){
            return __assign( store_method::MUL, e.__name() );
        }
        template<typename T>
        inline Elem & operator/=( const CompositeExp<T> & e){
            return __assign( store_method::DIV, e.__name() );
        }
    public :
        // d = src * s
        template<typename T>
        inline Elem & __assign( int op, const ScaleExp< ContainerExp<T>, double > &e ){
            func_impl::scale<Elem,T>( op, __self(), e.e.__name(), e.as_double() );
            return __self();
        }                
        template<typename T>
        inline Elem & __assign( const ScaleExp< CompositeExp<T>, double > &e ){
            __self()  = e.e.__name();
            __self() *= e.as_double();
            return __self();
        } 
        template<typename T>
        inline Elem & operator*=( const ScaleExp< CompositeExp<T>, double > &e ){
            __self() *= e.e.__name();
            __self() *= e.as_double();
            return __self();
        }
        template<typename T>
        inline Elem & operator/=( const ScaleExp< CompositeExp<T>, double > &e ){
            __self() /= e.e.__name();
            __self() /= e.as_double();
            return __self();
        }
        // d = a + b
        template<typename TA, typename TB>
        inline Elem & __assign( int op, const AddExp< ContainerExp<TA>, ContainerExp<TB> > &e ){
            func_impl::add<Elem,TA,TB>( op, __self(), e.a.__name(), e.b.__name() );
            return __self();
        }        
        
        
    };
};

namespace apex_exp_template{
    // e.R : a.R[k][i][j] = a[k][-i][-j]
    template<typename Elem>
    struct ReverseExp:public CompositeExp< ReverseExp<Elem> >{
        const Elem &e;
        ReverseExp( const Elem &exp ):e(exp){}
        // transpose of a transpose is orignal value
        inline const Elem & R() const{
            return e;
        }
    };

    // e.T 
    template<typename Elem>
    struct TransposeExp:public CompositeExp< TransposeExp<Elem> >{
        const Elem &e;
        TransposeExp( const Elem &exp ):e(exp){}
        // transpose of a transpose is orignal value
        inline const Elem & T() const{
            return e;
        }
    };

    // a + b 
    template<typename TA,typename TB>
    struct AddExp:public CompositeExp< AddExp<TA,TB> >{
        const TA &a;
        const TB &b;
        AddExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };  
    namespace operators{
        template<typename TA,typename TB,typename TAA, typename TBB>
        inline const AddExp<TAA,TBB> operator+( const Exp<TA,TAA> &a, const Exp<TB,TBB> &b ){
            return AddExp<TAA,TBB>( a.__alias(), b.__alias() );
        }
    };

    // a - b 
    template<typename TA,typename TB>
    struct SubExp:public CompositeExp< SubExp<TA,TB> >{
        const TA &a;
        const TB &b;
        SubExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };  
    namespace operators{
        template<typename TA,typename TB,typename TAA, typename TBB>
        inline const SubExp<TAA,TBB> operator-( const Exp<TA,TAA> &a, const Exp<TB,TBB> &b ){
            return SubExp<TAA,TBB>( a.__alias(), b.__alias() );
        }
    };

    // a * b 
    template<typename TA,typename TB>
    struct MulExp:public CompositeExp< MulExp<TA,TB> >{
        const TA &a;
        const TB &b;
        MulExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };  
    namespace operators{
        template<typename TA,typename TB,typename TAA, typename TBB>
        inline const MulExp<TAA,TBB> operator*( const Exp<TA,TAA> &a, const Exp<TB,TBB> &b ){
            return MulExp<TAA,TBB>( a.__alias(), b.__alias() );
        }
    };

    // a / b 
    template<typename TA,typename TB>
    struct DivExp:public CompositeExp< DivExp<TA,TB> >{
        const TA &a;
        const TB &b;
        DivExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };
    namespace operators{
        template<typename TA,typename TB,typename TAA, typename TBB>
        inline const DivExp<TAA,TBB> operator/( const Exp<TA,TAA> &a, const Exp<TB,TBB> &b ){
            return DivExp<TAA,TBB>( a.__alias(), b.__alias() );
        }
    };

    // dot(a,b)
    template<typename TA,typename TB>
    struct DotExp:public CompositeExp< DotExp<TA,TB> >{
        const TA &a;
        const TB &b;
        DotExp( const TA &ea, const TB &eb ):a(ea),b(eb){}
    };       
    namespace operators{
        template<typename TA,typename TB,typename TAA, typename TBB>
        inline const DotExp<TAA,TBB> dot( const Exp<TA,TAA> &a, const Exp<TB,TBB> &b ){
            return DotExp<TAA,TBB>( a.__alias(), b.__alias() );
        }
    };       

    // conv2(a,b,'V') conv2(a,b,'F') conv2(a,b,'E')
    template<typename TA,typename TB>
    struct Conv2Exp:public CompositeExp< Conv2Exp<TA,TB> >{
        const TA &a;
        const TB &b;
        char  op;
        Conv2Exp( const TA &ea, const TB &eb, char option ):a(ea),b(eb),op(option){}
    };       
    namespace operators{
        template<typename TA,typename TB,typename TAA, typename TBB>
        inline const Conv2Exp<TAA,TBB> conv2( const Exp<TA,TAA> &a, const Exp<TB,TBB> &b, char option ){
            return Conv2Exp<TAA,TBB>( a.__alias(), b.__alias(), option );
        }                        
    };       

    // e * s
    template<typename T,typename TV>
    struct ScaleExp:public CompositeExp< ScaleExp<T,TV> >{
        const T &e;
        TV s;
        ScaleExp( const T &exp, TV es ):e(exp),s(es){}
        inline float as_float()const{
            return (float)s;
        }
        inline double as_double()const{
            return (double)s;
        }
    };     

    namespace operators{
        // use double precision for safety concern
        template<typename T,typename TT>
        inline const ScaleExp<TT,double> operator*( const Exp<T,TT> &a, double s ){
            return ScaleExp<TT,double>( a.__alias(), s );
        }
        template<typename T,typename TT>
        inline const ScaleExp<TT,double> operator*( double s, const Exp<T,TT> &a ){
            return a * s;
        }
        template<typename T,typename TT>
        inline const ScaleExp<TT,double> operator/( const Exp<T,TT> &a, double s ){
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

