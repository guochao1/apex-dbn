#ifndef _APEX_TENSOR_GPU_H_
#define _APEX_TENSOR_GPU_H_

#include "apex_op_plan.h"
#include "apex_tensor_cpu.h"
#include "apex_tensor.h"


// data structure for tensor
namespace apex_tensor{
    struct GTensor1D{
        int           x_max;        
        size_t        pitch;
        TENSOR_FLOAT *elem;
        
        GTensor1D(){}
        GTensor1D( int x_max ){
            set_param( x_max ); 
        }        
        // set the parameter of current data
        inline void set_param( int x_max ){
            this->x_max = x_max;
        }        

        inline GTensor1D& operator =  ( TENSOR_FLOAT val );        
        inline GTensor1D& operator += ( TENSOR_FLOAT val );        
        inline GTensor1D& operator *= ( TENSOR_FLOAT val );        
        inline GTensor1D& operator += ( const GTensor1D &b );        
        inline GTensor1D& operator -= ( const GTensor1D &b );        

        inline apex_op_plan::TransposePlan<GTensor1D> T() const;
        inline GTensor1D& operator =  ( const apex_op_plan::ClonePlan      <CTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::AllocLikePlan  <CTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::SigmoidPlan      <GTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::SampleBinaryPlan <GTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::AddPlan <GTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::MulPlan <GTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::DotPlan  <GTensor1D,GTensor2D> &val );        
        inline GTensor1D& operator += ( const apex_op_plan::DotPlan  <GTensor1D,GTensor2D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::DotRTPlan<GTensor1D,GTensor2D> &val );        
        inline GTensor1D& operator += ( const apex_op_plan::DotRTPlan<GTensor1D,GTensor2D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::ScalePlan<GTensor1D,TENSOR_FLOAT> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::ScaleAddPlan<GTensor1D,TENSOR_FLOAT> &val );        
    };

    struct GTensor2D{
        int           x_max, y_max;        
        size_t        pitch;
        TENSOR_FLOAT *elem;

        GTensor2D(){}       
        GTensor2D( int y_max, int x_max ){
            set_param( y_max, x_max ); 
        }        
        // set the parameter of current data
        inline void set_param( int y_max, int x_max ){
            this->x_max = x_max;
            this->y_max = y_max;
        }        
        // operators
        inline       GTensor1D operator[]( int idx );
        inline const GTensor1D operator[]( int idx )const;
        inline GTensor2D& operator =  ( TENSOR_FLOAT val );
        inline GTensor2D& operator += ( TENSOR_FLOAT val );
        inline GTensor2D& operator *= ( TENSOR_FLOAT val );
        inline GTensor2D& operator += ( const GTensor2D &b );        
        inline GTensor2D& operator -= ( const GTensor2D &b );        

        inline apex_op_plan::TransposePlan<GTensor2D> T() const;
        inline GTensor2D& operator =  ( const apex_op_plan::ClonePlan      <CTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::AllocLikePlan  <CTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::SigmoidPlan      <GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::SampleBinaryPlan <GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::AddPlan <GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::MulPlan <GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::DotPlan  <GTensor2D,GTensor2D> &val );        
        inline GTensor2D& operator += ( const apex_op_plan::DotPlan  <GTensor2D,GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::DotRTPlan<GTensor2D,GTensor2D> &val );        
        inline GTensor2D& operator += ( const apex_op_plan::DotRTPlan<GTensor2D,GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::DotLTPlan<GTensor1D,GTensor1D> &val );        
        inline GTensor2D& operator += ( const apex_op_plan::DotLTPlan<GTensor1D,GTensor1D> &val );        
        inline GTensor2D& operator -= ( const apex_op_plan::DotLTPlan<GTensor1D,GTensor1D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::ScalePlan<GTensor2D,TENSOR_FLOAT> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::ScaleAddPlan<GTensor2D,TENSOR_FLOAT> &val );        
    };

    struct GTensor3D{
        int           x_max, y_max, z_max;                
        size_t        pitch;
        TENSOR_FLOAT *elem;
        GTensor3D(){}
        GTensor3D( int z_max, int y_max, int x_max ){
            set_param( z_max, y_max, x_max ); 
        }        
        // set the parameter of current data
        inline void set_param( int z_max, int y_max, int x_max ){
            this->x_max = x_max;
            this->y_max = y_max;
            this->z_max = z_max;
        }        
        // operators
        inline       GTensor2D operator[]( int idx );
        inline const GTensor2D operator[]( int idx )const;
        inline GTensor3D& operator =  ( TENSOR_FLOAT val );
        inline GTensor3D& operator += ( TENSOR_FLOAT val );
        inline GTensor3D& operator *= ( TENSOR_FLOAT val );
        inline GTensor3D& operator += ( const GTensor3D &b );        
        inline GTensor3D& operator -= ( const GTensor3D &b );        

        inline apex_op_plan::TransposePlan<GTensor3D> T() const;
        inline GTensor3D& operator =  ( const apex_op_plan::SigmoidPlan     <GTensor3D> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::SampleBinaryPlan<GTensor3D> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::AddPlan<GTensor3D> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::MulPlan<GTensor3D> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::ScalePlan<GTensor3D,TENSOR_FLOAT> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::ScaleAddPlan<GTensor3D,TENSOR_FLOAT> &val );        
    };
    
    struct GTensor4D{
        int           x_max, y_max, z_max, h_max;        
        size_t        pitch;

        TENSOR_FLOAT *elem;
        GTensor4D(){}
        GTensor4D( int h_max, int z_max, int y_max, int x_max ){
            set_param( h_max, z_max, y_max, x_max ); 
        }        
        // set the parameter of current data
        inline void set_param( int h_max, int z_max, int y_max, int x_max ){
            this->x_max = x_max;
            this->y_max = y_max;
            this->z_max = z_max;
            this->h_max = h_max;
        }        
        // operators
        inline       GTensor3D operator[]( int idx );
        inline const GTensor3D operator[]( int idx )const;
        inline GTensor4D& operator =  ( TENSOR_FLOAT val );
        inline GTensor4D& operator += ( TENSOR_FLOAT val );
        inline GTensor4D& operator *= ( TENSOR_FLOAT val );
        inline GTensor4D& operator += ( const GTensor4D &b );        
        inline GTensor4D& operator -= ( const GTensor4D &b );        

        inline apex_op_plan::TransposePlan<GTensor4D> T() const;
        inline GTensor4D& operator =  ( const apex_op_plan::SigmoidPlan<GTensor4D> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::SampleBinaryPlan<GTensor4D> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::AddPlan<GTensor4D> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::MulPlan<GTensor4D> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::ScalePlan<GTensor4D,TENSOR_FLOAT> &val );                        
        inline GTensor4D& operator =  ( const apex_op_plan::ScaleAddPlan<GTensor4D,TENSOR_FLOAT> &val );        
    };
    
    // inline functions for tensor
    
    // functions defined for tensor
    // intialize the tensor engine for use, seed is 
    // the seed for random number generator
    void init_tensor_engine_gpu( int seed );
    // this function is called when the program exits
    void destroy_tensor_engine_gpu();
 
    namespace tensor{
        // allocate space for given tensor
        void alloc_space( GTensor1D &ts );
        void alloc_space( GTensor2D &ts );
        void alloc_space( GTensor3D &ts );
        void alloc_space( GTensor4D &ts );
        
        // free space for given tensor
        void free_space( GTensor1D &ts );
        void free_space( GTensor2D &ts );
        void free_space( GTensor3D &ts );
        void free_space( GTensor4D &ts );
        
        // fill the tensor with real value
        void fill( GTensor1D &ts, TENSOR_FLOAT val );
        void fill( GTensor2D &ts, TENSOR_FLOAT val );
        void fill( GTensor3D &ts, TENSOR_FLOAT val );
        void fill( GTensor4D &ts, TENSOR_FLOAT val );
        
        // copy data from another tensor
        void copy( GTensor1D &dst, const CTensor1D &src );
        void copy( GTensor2D &dst, const CTensor2D &src );
        void copy( GTensor3D &dst, const CTensor3D &src );
        void copy( GTensor4D &dst, const CTensor4D &src );

        // copy data from another tensor
        void copy( GTensor1D &dst, const GTensor1D &src );
        void copy( GTensor2D &dst, const GTensor2D &src );
        void copy( GTensor3D &dst, const GTensor3D &src );
        void copy( GTensor4D &dst, const GTensor4D &src );
    };    
    
    //mapping functions 
    namespace tensor{
        void sigmoid( GTensor1D &mean, const GTensor1D &energy );
        void sigmoid( GTensor2D &mean, const GTensor2D &energy );
        void sigmoid( GTensor3D &mean, const GTensor3D &energy );
        void sigmoid( GTensor4D &mean, const GTensor4D &energy );
    };

    // sampling functions 
    namespace tensor{
        // sample binary distribution
        void sample_binary  ( GTensor1D &state, const GTensor1D &prob );
        void sample_binary  ( GTensor2D &state, const GTensor2D &prob );
        void sample_binary  ( GTensor3D &state, const GTensor3D &prob );
        void sample_binary  ( GTensor4D &state, const GTensor4D &prob );
        
        // sample gaussian distribution with certain sd
        void sample_gaussian( GTensor1D &state, const GTensor1D &mean, TENSOR_FLOAT sd );
        void sample_gaussian( GTensor2D &state, const GTensor2D &mean, TENSOR_FLOAT sd );
        void sample_gaussian( GTensor3D &state, const GTensor3D &mean, TENSOR_FLOAT sd );
        void sample_gaussian( GTensor4D &state, const GTensor4D &mean, TENSOR_FLOAT sd );
        
        // sample gaussian distribution with certain mean sd
        void sample_gaussian( GTensor1D &state, TENSOR_FLOAT sd );        
        void sample_gaussian( GTensor2D &state, TENSOR_FLOAT sd ); 
        void sample_gaussian( GTensor3D &state, TENSOR_FLOAT sd );        
        void sample_gaussian( GTensor4D &state, TENSOR_FLOAT sd );        
    };

    // arithmetic operations
    namespace tensor{
        // dst = a + b
        void add      ( GTensor1D &dst, const GTensor1D &a, const GTensor1D &b );
        void add      ( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );
        void add      ( GTensor3D &dst, const GTensor3D &a, const GTensor3D &b );
        void add      ( GTensor4D &dst, const GTensor4D &a, const GTensor4D &b );                
        // dst = a + b
        void mul      ( GTensor1D &dst, const GTensor1D &a, const GTensor1D &b );
        void mul      ( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );
        void mul      ( GTensor3D &dst, const GTensor3D &a, const GTensor3D &b );
        void mul      ( GTensor4D &dst, const GTensor4D &a, const GTensor4D &b );                
        // dst = a*sa + b*sb
        void scale_add( GTensor1D &dst, const GTensor1D &a, const GTensor1D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        void scale_add( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        void scale_add( GTensor3D &dst, const GTensor3D &a, const GTensor3D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        void scale_add( GTensor4D &dst, const GTensor4D &a, const GTensor4D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );                
        // dst = a - b
        void sub      ( GTensor1D &dst, const GTensor1D &a, const GTensor1D &b );
        void sub      ( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );
        void sub      ( GTensor3D &dst, const GTensor3D &a, const GTensor3D &b );
        void sub      ( GTensor4D &dst, const GTensor4D &a, const GTensor4D &b );
        // dst = a + val
        void add      ( GTensor1D &dst, const GTensor1D &a, TENSOR_FLOAT val );
        void add      ( GTensor2D &dst, const GTensor2D &a, TENSOR_FLOAT val );
        void add      ( GTensor3D &dst, const GTensor3D &a, TENSOR_FLOAT val );
        void add      ( GTensor4D &dst, const GTensor4D &a, TENSOR_FLOAT val );
        // dst = a * val
        void mul      ( GTensor1D &dst, const GTensor1D &a, TENSOR_FLOAT val );
        void mul      ( GTensor2D &dst, const GTensor2D &a, TENSOR_FLOAT val );
        void mul      ( GTensor3D &dst, const GTensor3D &a, TENSOR_FLOAT val );
        void mul      ( GTensor4D &dst, const GTensor4D &a, TENSOR_FLOAT val );

        // matrix multiplication
        // dst   = dot( a, b  ) 
        // note: the 1D tensor is treated as 1 * n matrix 
        void dot      ( GTensor1D &dst, const GTensor1D &a, const GTensor2D &b );    
        void dot      ( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );            
        // dst  += dot( a, b  ) 
        void add_dot  ( GTensor1D &dst, const GTensor1D &a, const GTensor2D &b );    
        void add_dot  ( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );                    
        // dst  = dot( a   ,  b.T )
        void dot_rt    ( GTensor1D &dst, const GTensor1D &a, const GTensor2D &b );    
        void dot_rt    ( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );    
        // dst += dot( a, b.T )
        void add_dot_rt    ( GTensor1D &dst, const GTensor1D &a, const GTensor2D &b );    
        void add_dot_rt    ( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );    
        // dst  = dot( a.T , b )
        void dot_lt    ( GTensor2D &dst, const GTensor1D &a, const GTensor1D &b );    
        void add_dot_lt( GTensor2D &dst, const GTensor1D &a, const GTensor1D &b );    
        void sub_dot_lt( GTensor2D &dst, const GTensor1D &a, const GTensor1D &b );    
    };
};

// definitions for inline functions 
#define TT1D GTensor1D
#define TT2D GTensor2D
#define TT3D GTensor3D
#define TT4D GTensor4D
#include "apex_tensor_inline.cpp"
#undef TT1D 
#undef TT2D 
#undef TT3D 
#undef TT4D 

#endif


