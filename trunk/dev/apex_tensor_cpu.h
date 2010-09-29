#ifndef _APEX_TENSOR_CPU_H_
#define _APEX_TENSOR_CPU_H_

/*!
 * \file apex_tensor_cpu.h
 * \brief CPU part implementation of tensor
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 */

#include "apex_tensor.h"
#include "apex_exp_template.h"
/*! \brief namespace of tensor imlementation */
namespace apex_tensor{
    using namespace apex_exp_template::operators;
};
namespace apex_tensor{
    /*! \brief pointer in CPU */
    template<typename TValue>
    class CPtr{
    private:
        /*! \brief real pointer */
        TValue *ptr;
    public:
        CPtr(){}
        /*! \brief constructor */
        CPtr( TValue *p ):ptr( p ){} 
        /*! \brief convert to real pointer */
        inline operator TValue*(){ return ptr; }
        /*! \brief convert to const real pointer */
        inline operator const TValue*() const{ return ptr; }
    };

    /*! \brief 1D tensor in CPU */
    class CTensor1D: public apex_exp_template::ContainerExp<CTensor1D>{
    public:
        /*! \brief number of element in x dimension */
        int x_max;
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
        inline void copy_param( const CTensor1D &exp );
        /*! \brief copy paramter from target */
        inline void copy_param( const GTensor1D &exp );
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
        unsigned int pitch_x;
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
        inline void copy_param( const CTensor2D &exp );
        /*! \brief copy paramter from target */
        inline void copy_param( const GTensor2D &exp );
        /*! \brief operator[] */
        inline CTensor1D operator[]( int idx );
        /*! \brief operator[] */
        inline const CTensor1D operator[]( int idx ) const;
    };

    /*! \brief 3D tensor in CPU */
    class CTensor3D: public apex_exp_template::ContainerExp<CTensor3D>{
    public:
        /*! \brief number of element in x dimension */
        int x_max;
        /*! \brief number of element in y dimension */
        int y_max;
        /*! \brief number of element in z dimension */
        int z_max;
        /*! \brief number of bytes allocated in x dimension */
        unsigned int pitch_x;
        /*! \brief number of bytes allocated in xy dimension */
        unsigned int pitch_xy;
        /*! \brief pointer to data */
        CPtr<TENSOR_FLOAT> elem;
        /*! \brief constructor */
        CTensor3D(){}
        /*! \brief constructor */
        CTensor3D( int z_max, int y_max, int x_max ){ 
            set_param( z_max, y_max, x_max ); 
        }        
        /*! \brief set parameters */
        inline void set_param( int z_max, int y_max,int x_max ){
            this->z_max = z_max;
            this->y_max = y_max;
            this->x_max = x_max;
        }
        /*! \brief copy paramter from target */
        inline void copy_param( const CTensor3D &exp );
        /*! \brief copy paramter from target */
        inline void copy_param( const GTensor3D &exp );
        /*! \brief operator[] */
        inline CTensor2D operator[]( int idx );
        /*! \brief operator[] */
        inline const CTensor2D operator[]( int idx ) const;
    };
    /*! \brief 4D tensor in CPU */
    class CTensor4D: public apex_exp_template::ContainerExp<CTensor4D>{
    public:
        /*! \brief number of element in x dimension */
        int x_max;
        /*! \brief number of element in y dimension */
        int y_max;
        /*! \brief number of element in z dimension */
        int z_max;
        /*! \brief number of element in h dimension */
        int h_max;
        /*! \brief number of bytes allocated in x dimension */
        unsigned int pitch_x;
        /*! \brief number of bytes allocated in xy dimension */
        unsigned int pitch_xy;
        /*! \brief number of bytes allocated in xyz dimension */
        unsigned int pitch_xyz;
        /*! \brief pointer to data */
        CPtr<TENSOR_FLOAT> elem;
        /*! \brief constructor */
        CTensor4D(){}
        /*! \brief constructor */
        CTensor4D( int h_max, int z_max, int y_max, int x_max ){ 
            set_param( h_max, z_max, y_max, x_max ); 
        }        
        /*! \brief set parameters */
        inline void set_param( int h_max, int z_max, int y_max, int x_max ){
            this->h_max = h_max;
            this->z_max = z_max;
            this->y_max = y_max;
            this->x_max = x_max;
        }
        /*! \brief copy paramter from target */
        inline void copy_param( const CTensor4D &exp );
        /*! \brief copy paramter from target */
        inline void copy_param( const GTensor4D &exp );
        /*! \brief operator[] */
        inline CTensor3D operator[]( int idx );
        /*! \brief operator[] */
        inline const CTensor3D operator[]( int idx ) const;
    };
};
#endif

