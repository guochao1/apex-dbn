#ifndef _APEX_EXP_TEMPLATE_H_
#define _APEX_EXP_TEMPLATE_H_
/*!
 * \file apex_exp_template.h
 * \brief expression template to do lazy evaluation
 * \author Tianqi Chen
 * \email  tqchen@apex.sjtu.edu.cn
 */

/*! \brief namespace of expression template */
namespace apex_exp_template{    
    /*! \brief namespace of operators defined in expression template 
     *  the user should using apex_exp_template::operators to enable all the operators
     */
    namespace operators{
    };
    /*! \brief namespace of solvers involved 
     *  the user should specialize the solver for each expressions
     */
    namespace solver_impl{
    };
    /*! \brief namespace of enumeration classes */
    namespace enums{
    };
};

namespace apex_exp_template{
    namespace enums{
        /*! \brief this class describes how to store result of calculation */
        template<typename Derived>
        class StoreMethod{
        protected:
            StoreMethod(){}
        };    
        /*! \brief operator= */
        class SaveTo: public StoreMethod<SaveTo> {
        private:
            SaveTo(){}
        public:
            const static SaveTo op;        
        };    
        /*! \brief singleton of SaveTo */
        const SaveTo SaveTo::op = SaveTo();
        
        /*! \brief operator+= */
        class AddTo : public StoreMethod<AddTo>{
        private:
            AddTo(){}
        public:
            const static AddTo  op;
        };
        /*! \brief singleton of AddTo */
        const AddTo AddTo::op = AddTo();
        
        /*! \brief operator-= */
        class SubTo : public StoreMethod<SubTo>{
        private:
            SubTo(){}
        public:
            const static SubTo  op;
        };
        /*! \brief singleton of SubTo */
        const SubTo SubTo::op = SubTo();
        
        /*! \brief operator*= */
        class MulTo : public StoreMethod<MulTo>{
        private:
            MulTo(){}
        public:
            const static MulTo  op;
        };
        /*! \brief singleton of MulTo */
        const MulTo MulTo::op = MulTo();
        
        /*! \brief operator/= */
        class DivTo : public StoreMethod<DivTo>{
        private:
            DivTo(){}
        public:
            const static DivTo  op;
        };
        /*! \brief singleton of MulTo */
        const DivTo DivTo::op = DivTo();
    };
};

namespace apex_exp_template{
    template<typename Elem>
    class TransposeExp;
    template<typename Elem>
    class ReverseExp;

    /*! \brief base expression class of all expressions */
    template<typename Name, typename Alias>
    class Exp{
    protected:
        Exp(){}
    public:
        /*! 
         *\brief evaluation src and store to dst 
         *\param st storage method
         *\param dst destination to be stored
         *\param src source expression
         *\sa StoreMethod
         */ 
        template<typename ST, typename Dst>
        inline void __eval( const enums::StoreMethod<ST> &st, Dst &dst, const Name &src ) const; 
        /*! 
         *\brief  return true derived class 
         *\return true derived class
         */
        inline Name &__name(){
            return *static_cast<Name*>(this);
        }
        /*! 
         *\brief  return true derived class 
         *\return true derived class
         */
        inline const Name &__name_const() const{
            return *static_cast<const Name*>(this);
        }
        /*! 
         *\brief  return alias derived class 
         * alias is used to identify the type of each class in composite
         *\return true derived class
         *\sa ContainerExp CompositeExp
         */
        inline const Alias & __alias_const() const{
            return *static_cast<const Alias*>(this);
        }   
        /*! 
         *\brief transpose of a matrix
         *\return transpose of current expression
         */
        inline const TransposeExp<Alias> T() const{
            return TransposeExp<Alias>( this->__alias_const() );
        }
        /*! 
         *\brief reverse of last two dimensions of matrix d.R[k][i][j] = d[k][-i][-j]
         *\return reverse of current expression
         */
        inline const ReverseExp<Alias> R() const{
            return ReverseExp<Alias>( this->__alias_const() );
        }
    };
    /*! \brief base class of all composite expressions
     *  this is a alias class, we use it to identify difference between 
     *  basic variables and composite expressions
     * \sa ContainerExp
     */
    template<typename Derived>
    class CompositeExp: public Exp< Derived, CompositeExp<Derived> >{
    protected:
        CompositeExp(){}
    };
    /*! \brief base class of all variables
     *  this is a alias class, we use it to identify difference between 
     *  basic variables and composite expressions
     * \sa CompositeExp
     */
    template<typename Derived>
    class ContainerExp: public Exp< Derived, ContainerExp<Derived> >{
    protected:
        ContainerExp(){}
    public   :                
        /*! \brief implementation of operator= */
        template<typename Name,typename Alias>
        inline Derived &__assign( const Exp<Name,Alias> &exp ){
            exp.__name_const().__eval( enums::SaveTo::op, this->__name(), exp.__name_const() );
            return this->__name();
        }
        /*! \brief implementation of operator+= */
        template<typename Name,typename Alias>
        inline Derived &operator+=( const Exp<Name,Alias> &exp ){
            exp.__name_const().__eval( enums::AddTo::op, this->__name(), exp.__name_const() );
            return this->__name();
        }
        /*! \brief implementation of operator-= */
        template<typename Name,typename Alias>
        inline Derived &operator-=( const Exp<Name,Alias> &exp ){
            exp.__name_const().__eval( enums::SubTo::op, this->__name(), exp.__name_const() );
            return this->__name();
        }
        /*! \brief implementation of operator*= */
        template<typename Name,typename Alias>
        inline Derived &operator*=( const Exp<Name,Alias> &exp ){
            exp.__name_const().__eval( enums::MulTo::op, this->__name(), exp.__name_const() );
            return this->__name();
        }
        /*! \brief implementation of operator/= */
        template<typename Name,typename Alias>
        inline Derived &operator/=( const Exp<Name,Alias> &exp ){
            exp.__name_const().__eval( enums::DivTo::op, this->__name(), exp.__name_const() );
            return this->__name();
        }        
    };            
};

namespace apex_exp_template{
    /*! transpose of a expression*/
    template<typename Elem>
    class TransposeExp: public CompositeExp< TransposeExp<Elem> >{
    public:
        const Elem &exp;
        TransposeExp( const Elem &e ):exp(e){}
        inline const Elem & T() const{
            return exp;
        }
    };
    /*! transpose of a expression*/
    template<typename Elem>
    class ReverseExp: public CompositeExp< TransposeExp<Elem> >{
    public:
        const Elem &exp;
        ReverseExp( const Elem &e ):exp(e){}
        inline const Elem & R() const{
            return exp;
        }
    };
};

namespace apex_exp_template{
    /*!
     *\brief base class of map type operators 
     * map type operators map one operation to another
     */
    template<typename Derived,typename Src>
    class MapExp: public CompositeExp<Derived>{        
    public:
        const Src &exp;
        MapExp( const Src &e ):exp(e){} 
        /*! \brief rule specialization, in a map chain, use dst as temporary storage */
        template<typename Dst, typename T>
        inline void __eval( const enums::SaveTo &s, Dst &dst, const MapExp< CompositeExp<T>, Src > &src ) const{
            dst = src.exp;
            dst = Derived( ContainerExp<Dst>(dst) );
        }
    };
};

namespace apex_exp_template{
    namespace solver_impl{
        /*! 
         * \brief solver interface to solve scale problem 
         * user must specialize the class to create specific solvers of types to support
         */
        template<typename ST,typename T, typename TV>
        struct ScaleSolver{
            /*! \brief implement dst = src*scalar */
            static inline void eval( T &dst, const T &src, TV scalar );
        };
    };
    /*! \brief scale expression which represent exp* scalar */
    template<typename Elem,typename TValue>
    class ScaleExp: public MapExp< ScaleExp<Elem,TValue>, Elem >{
    public:        
        TValue scalar;
        ScaleExp( const Elem &e, TValue s ):MapExp<ScaleExp<Elem,TValue>, Elem>(e),scalar(s){}
        /*! \brief rule specialization, combine scale together */
        template<typename Dst, typename T>
        inline void __eval( const enums::SaveTo &s, Dst &dst, const ScaleExp< CompositeExp< ScaleExp<T,double> >, double > &src ) const{
            dst = ScaleExp<T,double>( src.exp.__name_const().exp, src.scalar * src.exp.__name_const().scalar );
        }        
        /*! \brief basic specialization of scale */
        template<typename ST,typename T,typename TV>
        inline void __eval( const enums::StoreMethod<ST> &s, T &dst, const ScaleExp< ContainerExp<T>, TV > &src ) const{
            solver_impl::ScaleSolver<ST,T,TV>::eval( dst, src.exp.__name_const(), src.scalar );
        }
    };
    namespace operators{
        /*! \brief operator overload for scale */
        template<typename T,typename TT>
        inline ScaleExp<TT,double> operator*( const Exp<T,TT> &exp, double scalar ){
            return ScaleExp<TT,double>( exp.__alias_const(), scalar ); 
        }
        /*! \brief operator overload for scale */
        template<typename T,typename TT>
        inline ScaleExp<TT,double> operator*( double scalar, const Exp<T,TT> &exp ){
            return exp *  scalar;
        }
        /*! \brief operator overload for scale */
        template<typename T,typename TT>
        inline ScaleExp<TT,double> operator/( const Exp<T,TT> &exp, double scalar ){
            return ScaleExp<TT,double>( exp.__alias_const(), 1.0/scalar ); 
        }
    };
};
#endif
