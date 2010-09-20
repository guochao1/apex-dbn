#ifndef _APEX_TENSOR_CPU_H_
#define _APEX_TENSOR_CPU_H_

/*!
 * \file apex_tensor_cpu.h
 * \brief CPU part implementation of tensor
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 */

#include "apex_tensor.h"
#include "apex_exp_template.h"

namespace apex_tensor{
    /*! \brief pointer in CPU */
    template<typename TValue>
    class CPtr{
    private:
        /*! \brief real pointer */
        TValue *ptr;
    public:
        /*! \brief constructor */
        CPtr( TValue *p ):ptr( p ){} 
        /*! \brief convert to real pointer */
        inline operator TValue*(){
            return ptr;
        }
        /*! \brief convert to const real pointer */
        inline operator const TValue*()const{
            return ptr;
        }
    };

    /*! \brief 1D tensor in CPU */
    class CTensor1D: public apex_exp_template::ContainerExp<CTensor1D>{
    public:
        /*! \brief number of element in x dimension */
        int x_max;
        /*! \brief number of bytes allocated in x dimension */
        unsigned int pitch;
        /*! \brief pointer to data */
        CPtr<TENSOR_FLOAT> elem;
        /*! \brief constructor */
        CTensor1D(){}
        /*! \brief constructor */
        CTensor1D( int x_max ){ 
            set_param( x_max ); 
        }        
        /*! \brief set parameters */
        inline void set_param( int x_max ){
            this->x_max = x_max;
        }
        /*! \brief copy paramter from target */
        inline void copy_param( const CTensor1D &exp ){
            this->x_max = exp.x_max;
        }
        /*! \brief copy paramter from target */
        inline void copy_param( const GTensor1D &exp ){
            this->x_max = exp.x_max;
        }
        /*! \brief operator[] */
        inline TENSOR_FLOAT& operator[]( int idx ){
            return elem[idx];
        }        
        /*! \brief operator[] */
        inline const TENSOR_FLOAT& operator[]( int idx )const{
            return elem[idx];
        }    
    };

    /*! \brief 2D tensor in CPU */
    class CTensor2D: public apex_exp_template::ContainerExp<CTensor2D>{
    public:
        /*! \brief number of element in x dimension */
        int x_max;
        /*! \brief number of element in y dimension */
        int y_max;
        /*! \brief number of bytes allocated in x dimension */
        unsigned int pitch;
        /*! \brief pointer to data */
        CPtr<TENSOR_FLOAT> elem;
        /*! \brief constructor */
        CTensor2D(){}
        /*! \brief constructor */
        CTensor2D( int y_max, int x_max ){ 
            set_param( y_max, x_max ); 
        }        
        /*! \brief set parameters */
        inline void set_param( int y_max,int x_max ){
            this->y_max = y_max;
            this->x_max = x_max;
        }
        /*! \brief copy paramter from target */
        inline void copy_param( const CTensor2D &exp ){
            this->y_max = exp.y_max;
            this->x_max = exp.x_max;
        }
        /*! \brief copy paramter from target */
        inline void copy_param( const GTensor2D &exp ){
            this->y_max = exp.y_max;
            this->x_max = exp.x_max;
        }        
    };
};

#endif

