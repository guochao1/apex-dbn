#ifndef _APEX_TENSOR_GPU_H_
#define _APEX_TENSOR_GPU_H_

/*!
 * \file apex_tensor_gpu.h
 * \brief GPU part implementation of tensor
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 */

#include "apex_tensor.h"
#include "apex_exp_template.h"

/*! \brief namespace of tensor imlementation */
namespace apex_tensor{
    using namespace apex_exp_template::operators;
};
namespace apex_tensor{
    /*! \brief pointer in GPU */
    template<typename TValue>
    class GPtr{
    private:
        /*! \brief real pointer */
        TValue *ptr;
    public:
        GPtr(){}
        /*! \brief constructor */
        GPtr( TValue *p ):ptr( p ){} 
        /*! \brief convert to real pointer */
        inline operator TValue*(){
            return ptr;
        }
        /*! \brief convert to const real pointer */
        inline operator const TValue*()const{
            return ptr;
        }
    };

    /*! \brief 1D tensor in GPU */
    class GTensor1D: public apex_exp_template::ContainerExp<GTensor1D>{
    public:
        /*! \brief number of element in x dimension */
        int x_max;
        /*! \brief number of bytes allocated in x dimension */
        unsigned int pitch;
        /*! \brief pointer to data */
        GPtr<TENSOR_FLOAT> elem;
        /*! \brief constructor */
        GTensor1D(){}
        /*! \brief constructor */
        GTensor1D( int x_max ){ 
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
    };

    /*! \brief 2D tensor in GPU */
    class GTensor2D: public apex_exp_template::ContainerExp<GTensor2D>{
    public:
        /*! \brief number of element in x dimension */
        int x_max;
        /*! \brief number of element in y dimension */
        int y_max;
        /*! \brief number of bytes allocated in x dimension */
        unsigned int pitch;
        /*! \brief pointer to data */
        GPtr<TENSOR_FLOAT> elem;
        /*! \brief constructor */
        GTensor2D(){}
        /*! \brief constructor */
        GTensor2D( int y_max, int x_max ){ 
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
        inline GTensor1D operator[]( int idx );
        /*! \brief operator[] */
        inline const GTensor1D operator[]( int idx ) const;
    };

    /*! \brief 3D tensor in GPU */
    class GTensor3D: public apex_exp_template::ContainerExp<GTensor3D>{
    public:
        /*! \brief number of element in x dimension */
        int x_max;
        /*! \brief number of element in y dimension */
        int y_max;
        /*! \brief number of element in z dimension */
        int z_max;
        /*! \brief number of bytes allocated in x dimension */
        unsigned int pitch;
        /*! \brief pointer to data */
        GPtr<TENSOR_FLOAT> elem;
        /*! \brief constructor */
        GTensor3D(){}
        /*! \brief constructor */
        GTensor3D( int z_max, int y_max, int x_max ){ 
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
        inline GTensor2D operator[]( int idx );
        /*! \brief operator[] */
        inline const GTensor2D operator[]( int idx ) const;
    };
    /*! \brief 4D tensor in GPU */
    class GTensor4D: public apex_exp_template::ContainerExp<GTensor4D>{
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
        unsigned int pitch;
        /*! \brief pointer to data */
        GPtr<TENSOR_FLOAT> elem;
        /*! \brief constructor */
        GTensor4D(){}
        /*! \brief constructor */
        GTensor4D( int h_max, int z_max, int y_max, int x_max ){ 
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
        inline GTensor3D operator[]( int idx );
        /*! \brief operator[] */
        inline const GTensor3D operator[]( int idx ) const;
    };
};
#endif

