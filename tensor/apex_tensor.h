#ifndef _APEX_TENSOR_H_
#define _APEX_TENSOR_H_


#include <cstdio>
#include <cmath>
#include "apex_op_plan.h"

// data structure for tensor
namespace apex_tensor{
    // defines the type of elements in tensor
    typedef float TENSOR_FLOAT;

    struct Tensor2D;

    struct Tensor1D{
        size_t        x_max;        
        size_t        pitch;
        TENSOR_FLOAT *elem;
        
        Tensor1D(){}
        Tensor1D( size_t x_max ){
            set_param( x_max ); 
        }        
        // set the parameter of current data
        inline void set_param( size_t x_max ){
            this->x_max = x_max;
        }        
        // operators       
        inline TENSOR_FLOAT& operator[]( int idx ){
            return elem[idx];
        }        
        inline const TENSOR_FLOAT& operator[]( int idx )const{
            return elem[idx];
        }    

        inline Tensor1D& operator =  ( TENSOR_FLOAT val );        
        inline Tensor1D& operator += ( TENSOR_FLOAT val );        
        inline Tensor1D& operator *= ( TENSOR_FLOAT val );        
        inline Tensor1D& operator += ( const Tensor1D &b );        
        inline Tensor1D& operator -= ( const Tensor1D &b );        
        
        inline Tensor1D& operator =  ( const apex_op_plan::AddPlan<Tensor1D> &val );        
        inline Tensor1D& operator =  ( const apex_op_plan::DotPlan<Tensor2D,Tensor1D> &val );        
        inline Tensor1D& operator =  ( const apex_op_plan::ScalePlan<Tensor1D,TENSOR_FLOAT> &val );        
        inline Tensor1D& operator =  ( const apex_op_plan::ScaleAddPlan<Tensor1D,TENSOR_FLOAT> &val );        
    };

    struct Tensor2D{
        size_t        x_max, y_max;        
        size_t        pitch;
        TENSOR_FLOAT *elem;

        Tensor2D(){}       
        Tensor2D( size_t x_max, size_t y_max ){
            set_param( x_max, y_max ); 
        }        
        // set the parameter of current data
        inline void set_param( size_t x_max, size_t y_max ){
            this->x_max = x_max;
            this->y_max = y_max;
        }        
        // operators
        inline       Tensor1D operator[]( int idx );
        inline const Tensor1D operator[]( int idx )const;
        inline Tensor2D& operator =  ( TENSOR_FLOAT val );
        inline Tensor2D& operator += ( TENSOR_FLOAT val );
        inline Tensor2D& operator *= ( TENSOR_FLOAT val );
        inline Tensor2D& operator += ( const Tensor2D &b );        
        inline Tensor2D& operator -= ( const Tensor2D &b );        

        inline Tensor2D& operator =  ( const apex_op_plan::AddPlan<Tensor2D> &val );        
        inline Tensor2D& operator =  ( const apex_op_plan::DotPlan<Tensor2D,Tensor2D> &val );        
        inline Tensor2D& operator =  ( const apex_op_plan::ScalePlan<Tensor2D,TENSOR_FLOAT> &val );        
        inline Tensor2D& operator =  ( const apex_op_plan::ScaleAddPlan<Tensor2D,TENSOR_FLOAT> &val );        
    };

    struct Tensor3D{
        size_t        x_max, y_max, z_max;                
        size_t        pitch;
        TENSOR_FLOAT *elem;
        Tensor3D(){}
        Tensor3D( size_t x_max, size_t y_max, size_t z_max ){
            set_param( x_max, y_max, z_max ); 
        }        
        // set the parameter of current data
        inline void set_param( size_t x_max, size_t y_max, size_t z_max ){
            this->x_max = x_max;
            this->y_max = y_max;
            this->z_max = z_max;
        }        
        // operators
        inline       Tensor2D operator[]( int idx );
        inline const Tensor2D operator[]( int idx )const;
        inline Tensor3D& operator =  ( TENSOR_FLOAT val );
        inline Tensor3D& operator += ( TENSOR_FLOAT val );
        inline Tensor3D& operator *= ( TENSOR_FLOAT val );
        inline Tensor3D& operator += ( const Tensor3D &b );        
        inline Tensor3D& operator -= ( const Tensor3D &b );        
        
        inline Tensor3D& operator =  ( const apex_op_plan::AddPlan<Tensor3D> &val );        
        inline Tensor3D& operator =  ( const apex_op_plan::ScalePlan<Tensor3D,TENSOR_FLOAT> &val );        
        inline Tensor3D& operator =  ( const apex_op_plan::ScaleAddPlan<Tensor3D,TENSOR_FLOAT> &val );        
    };
    
    struct Tensor4D{
        size_t        x_max, y_max, z_max, h_max;        
        size_t        pitch;

        TENSOR_FLOAT *elem;
        Tensor4D(){}
        Tensor4D( size_t x_max, size_t y_max, size_t z_max, size_t h_max ){
            set_param( x_max, y_max, z_max, h_max ); 
        }        
        // set the parameter of current data
        inline void set_param( size_t x_max, size_t y_max, size_t z_max, size_t h_max ){
            this->x_max = x_max;
            this->y_max = y_max;
            this->z_max = z_max;
            this->h_max = h_max;
        }        
        // operators
        inline       Tensor3D operator[]( int idx );
        inline const Tensor3D operator[]( int idx )const;
        inline Tensor4D& operator =  ( TENSOR_FLOAT val );
        inline Tensor4D& operator += ( TENSOR_FLOAT val );
        inline Tensor4D& operator *= ( TENSOR_FLOAT val );
        inline Tensor4D& operator += ( const Tensor4D &b );        
        inline Tensor4D& operator -= ( const Tensor4D &b );        

        inline Tensor4D& operator =  ( const apex_op_plan::AddPlan<Tensor4D> &val );        
        inline Tensor4D& operator =  ( const apex_op_plan::ScalePlan<Tensor4D,TENSOR_FLOAT> &val );        
        inline Tensor4D& operator =  ( const apex_op_plan::ScaleAddPlan<Tensor4D,TENSOR_FLOAT> &val );        
    };
    
    // inline functions for tensor
    
    // functions defined for tensor
    namespace tensor{
        // allocate space for given tensor
        void alloc_space( Tensor1D &ts );
        void alloc_space( Tensor2D &ts );
        void alloc_space( Tensor3D &ts );
        void alloc_space( Tensor4D &ts );
        
        // free space for given tensor
        void free_space( Tensor1D &ts );
        void free_space( Tensor2D &ts );
        void free_space( Tensor3D &ts );
        void free_space( Tensor4D &ts );
        
        // fill the tensor with real value
        void fill( Tensor1D &ts, TENSOR_FLOAT val );
        void fill( Tensor2D &ts, TENSOR_FLOAT val );
        void fill( Tensor3D &ts, TENSOR_FLOAT val );
        void fill( Tensor4D &ts, TENSOR_FLOAT val );
        
        // save tensor to file
        void save_to_file( const Tensor1D &ts, FILE *dst );
        void save_to_file( const Tensor2D &ts, FILE *dst );
        void save_to_file( const Tensor3D &ts, FILE *dst );
        void save_to_file( const Tensor4D &ts, FILE *dst );      
        
        // load tensor from file 
        void load_from_file( Tensor1D &ts, FILE *src );
        void load_from_file( Tensor2D &ts, FILE *src );
        void load_from_file( Tensor3D &ts, FILE *src );
        void load_from_file( Tensor4D &ts, FILE *src );      
        
        // copy data from another tensor
        void copy( Tensor1D &dst, const Tensor1D &src );
        void copy( Tensor2D &dst, const Tensor2D &src );
        void copy( Tensor3D &dst, const Tensor3D &src );
        void copy( Tensor4D &dst, const Tensor4D &src );
    };    
    
    //mapping functions 
    namespace tensor{
        void sigmoid( Tensor1D &mean, const Tensor1D &energy );
        void sigmoid( Tensor2D &mean, const Tensor2D &energy );
        void sigmoid( Tensor3D &mean, const Tensor3D &energy );
        void sigmoid( Tensor4D &mean, const Tensor4D &energy );
    };

    // sampling functions 
    namespace tensor{
        // sample binary distribution
        void sample_binary  ( Tensor1D &state, const Tensor1D &prob );
        void sample_binary  ( Tensor2D &state, const Tensor2D &prob );
        void sample_binary  ( Tensor3D &state, const Tensor3D &prob );
        void sample_binary  ( Tensor4D &state, const Tensor4D &prob );
        
        // sample gaussian distribution with certain sd
        void sample_gaussian( Tensor1D &state, const Tensor1D &mean, float sd );
        void sample_gaussian( Tensor2D &state, const Tensor2D &mean, float sd );
        void sample_gaussian( Tensor3D &state, const Tensor3D &mean, float sd );
        void sample_gaussian( Tensor4D &state, const Tensor4D &mean, float sd );
        
        // sample gaussian distribution with certain mean sd
        void sample_gaussian( Tensor1D &state, float sd );        
        void sample_gaussian( Tensor2D &state, float sd ); 
        void sample_gaussian( Tensor3D &state, float sd );        
        void sample_gaussian( Tensor4D &state, float sd );        
    };

    // arithmetic operations
    namespace tensor{
        // dst = a + b
        void add      ( Tensor1D &dst, const Tensor1D &a, const Tensor1D &b );
        void add      ( Tensor2D &dst, const Tensor2D &a, const Tensor2D &b );
        void add      ( Tensor3D &dst, const Tensor3D &a, const Tensor3D &b );
        void add      ( Tensor4D &dst, const Tensor4D &a, const Tensor4D &b );                
        // dst = a*sa + b*sb
        void scale_add( Tensor1D &dst, const Tensor1D &a, const Tensor1D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        void scale_add( Tensor2D &dst, const Tensor2D &a, const Tensor2D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        void scale_add( Tensor3D &dst, const Tensor3D &a, const Tensor3D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        void scale_add( Tensor4D &dst, const Tensor4D &a, const Tensor4D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );                
        // dst = a - b
        void sub      ( Tensor1D &dst, const Tensor1D &a, const Tensor1D &b );
        void sub      ( Tensor2D &dst, const Tensor2D &a, const Tensor2D &b );
        void sub      ( Tensor3D &dst, const Tensor3D &a, const Tensor3D &b );
        void sub      ( Tensor4D &dst, const Tensor4D &a, const Tensor4D &b );
        // dst = a + val
        void add      ( Tensor1D &dst, const Tensor1D &a, TENSOR_FLOAT val );
        void add      ( Tensor2D &dst, const Tensor2D &a, TENSOR_FLOAT val );
        void add      ( Tensor3D &dst, const Tensor3D &a, TENSOR_FLOAT val );
        void add      ( Tensor4D &dst, const Tensor4D &a, TENSOR_FLOAT val );
        // dst = a * val
        void mul      ( Tensor1D &dst, const Tensor1D &a, TENSOR_FLOAT val );
        void mul      ( Tensor2D &dst, const Tensor2D &a, TENSOR_FLOAT val );
        void mul      ( Tensor3D &dst, const Tensor3D &a, TENSOR_FLOAT val );
        void mul      ( Tensor4D &dst, const Tensor4D &a, TENSOR_FLOAT val );

        // matrix multiplication
        // dst  = dot( a, b  ) 
        void dot      ( Tensor1D &dst, const Tensor2D a, const Tensor1D &b );    
        void dot      ( Tensor2D &dst, const Tensor2D a, const Tensor2D &b );            
               
        // dst  = dot( a.T, b)
        void dot_t    ( Tensor1D &dst, const Tensor2D a, const Tensor1D &b );    
        void dot_t    ( Tensor2D &dst, const Tensor2D a, const Tensor2D &b );    
        void dot_t    ( Tensor2D &dst, const Tensor1D a, const Tensor1D &b );    

    };
};

// definitions for inline functions 
#define TT1D Tensor1D
#define TT2D Tensor2D
#define TT3D Tensor3D
#define TT4D Tensor4D
#include "apex_tensor_inline.cpp"
#undef TT1D 
#undef TT2D 
#undef TT3D 
#undef TT4D 

#endif


