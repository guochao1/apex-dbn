#ifndef _APEX_TENSOR_CPU_H_
#define _APEX_TENSOR_CPU_H_

#include "apex_op_plan.h"
#include "apex_tensor.h"

// data structure for tensor
namespace apex_tensor{
    struct CTensor1D{
        int           x_max;        
        size_t        pitch;
        TENSOR_FLOAT *elem;
        
        CTensor1D(){}
        CTensor1D( int x_max ){
            set_param( x_max ); 
        }        
        // set the parameter of current data
        inline void set_param( int x_max ){
            this->x_max = x_max;
        }        
        // operators       
        inline TENSOR_FLOAT& operator[]( int idx ){
            return elem[idx];
        }        
        inline const TENSOR_FLOAT& operator[]( int idx )const{
            return elem[idx];
        }    
                
        inline CTensor1D& operator =  ( TENSOR_FLOAT val );        
        inline CTensor1D& operator += ( TENSOR_FLOAT val );        
        inline CTensor1D& operator *= ( TENSOR_FLOAT val );        
        inline CTensor1D& operator += ( const CTensor1D &b );        
        inline CTensor1D& operator -= ( const CTensor1D &b );        
        
        inline apex_op_plan::TransposePlan<CTensor1D> T() const;
        inline CTensor1D& operator =  ( const apex_op_plan::ClonePlan         <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::AllocLikePlan     <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::AllocLikePlan     <GTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::SigmoidPlan       <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::SampleBinaryPlan  <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::AddPlan  <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::MulPlan  <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::DotPlan  <CTensor1D,CTensor2D> &val );        
        inline CTensor1D& operator += ( const apex_op_plan::DotPlan  <CTensor1D,CTensor2D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::DotRTPlan<CTensor1D,CTensor2D> &val );        
        inline CTensor1D& operator += ( const apex_op_plan::DotRTPlan<CTensor1D,CTensor2D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::ScalePlan<CTensor1D,TENSOR_FLOAT> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::ScaleAddPlan<CTensor1D,TENSOR_FLOAT> &val );        
    };

    struct CTensor2D{
        int           x_max, y_max;        
        size_t        pitch;
        TENSOR_FLOAT *elem;

        CTensor2D(){}       
        CTensor2D( int y_max, int x_max ){
            set_param( y_max, x_max ); 
        }        
        // set the parameter of current data
        inline void set_param( int y_max, int x_max ){
            this->x_max = x_max;
            this->y_max = y_max;
        }        
        // operators
        inline       CTensor1D operator[]( int idx );
        inline const CTensor1D operator[]( int idx )const;
        inline CTensor2D& operator =  ( TENSOR_FLOAT val );
        inline CTensor2D& operator += ( TENSOR_FLOAT val );
        inline CTensor2D& operator *= ( TENSOR_FLOAT val );
        inline CTensor2D& operator += ( const CTensor2D &b );        
        inline CTensor2D& operator -= ( const CTensor2D &b );        

        inline apex_op_plan::TransposePlan<CTensor2D> T() const;
        inline CTensor2D& operator =  ( const apex_op_plan::ClonePlan         <CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::AllocLikePlan     <CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::AllocLikePlan     <GTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::SigmoidPlan       <CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::SampleBinaryPlan  <CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::AddPlan  <CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::MulPlan  <CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::DotPlan  <CTensor2D,CTensor2D> &val );        
        inline CTensor2D& operator += ( const apex_op_plan::DotPlan  <CTensor2D,CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::DotRTPlan<CTensor2D,CTensor2D> &val );        
        inline CTensor2D& operator += ( const apex_op_plan::DotRTPlan<CTensor2D,CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::DotLTPlan<CTensor1D,CTensor1D> &val );        
        inline CTensor2D& operator += ( const apex_op_plan::DotLTPlan<CTensor1D,CTensor1D> &val );        
        inline CTensor2D& operator -= ( const apex_op_plan::DotLTPlan<CTensor1D,CTensor1D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::ScalePlan<CTensor2D,TENSOR_FLOAT> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::ScaleAddPlan<CTensor2D,TENSOR_FLOAT> &val );        
    };

    struct CTensor3D{
        int           x_max, y_max, z_max;                
        size_t        pitch;
        TENSOR_FLOAT *elem;
        CTensor3D(){}
        CTensor3D( int z_max, int y_max, int x_max ){
            set_param( z_max, y_max, x_max ); 
        }        
        // set the parameter of current data
        inline void set_param( int z_max, int y_max, int x_max ){
            this->x_max = x_max;
            this->y_max = y_max;
            this->z_max = z_max;
        }        
        // operators
        inline       CTensor2D operator[]( int idx );
        inline const CTensor2D operator[]( int idx )const;
        inline CTensor3D& operator =  ( TENSOR_FLOAT val );
        inline CTensor3D& operator += ( TENSOR_FLOAT val );
        inline CTensor3D& operator *= ( TENSOR_FLOAT val );
        inline CTensor3D& operator += ( const CTensor3D &b );        
        inline CTensor3D& operator -= ( const CTensor3D &b );        
        
        inline apex_op_plan::TransposePlan<CTensor3D> T() const;
        inline CTensor3D& operator =  ( const apex_op_plan::ClonePlan         <CTensor3D> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::AllocLikePlan     <CTensor3D> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::AllocLikePlan     <GTensor3D> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::SigmoidPlan       <CTensor3D> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::SampleBinaryPlan  <CTensor3D> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::AddPlan<CTensor3D> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::MulPlan<CTensor3D> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::ScalePlan<CTensor3D,TENSOR_FLOAT> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::ScaleAddPlan<CTensor3D,TENSOR_FLOAT> &val );        
    };
    
    struct CTensor4D{
        int           x_max, y_max, z_max, h_max;        
        size_t        pitch;

        TENSOR_FLOAT *elem;
        CTensor4D(){}
        CTensor4D( int h_max, int z_max, int y_max, int x_max ){
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
        inline       CTensor3D operator[]( int idx );
        inline const CTensor3D operator[]( int idx )const;
        inline CTensor4D& operator =  ( TENSOR_FLOAT val );
        inline CTensor4D& operator += ( TENSOR_FLOAT val );
        inline CTensor4D& operator *= ( TENSOR_FLOAT val );
        inline CTensor4D& operator += ( const CTensor4D &b );        
        inline CTensor4D& operator -= ( const CTensor4D &b );        

        inline apex_op_plan::TransposePlan<CTensor4D> T() const;
        inline CTensor4D& operator =  ( const apex_op_plan::ClonePlan       <CTensor4D> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::AllocLikePlan   <CTensor4D> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::AllocLikePlan   <GTensor4D> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::SigmoidPlan     <CTensor4D> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::SampleBinaryPlan<CTensor4D> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::AddPlan<CTensor4D> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::MulPlan<CTensor4D> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::ScalePlan<CTensor4D,TENSOR_FLOAT> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::ScaleAddPlan<CTensor4D,TENSOR_FLOAT> &val );        
    };
    
    // inline functions for tensor
    
    // functions defined for tensor

    // intialize the tensor engine for use, seed is 
    // the seed for random number generator
    void init_tensor_engine_cpu( int seed );
    // this function is called when the program exits
    void destroy_tensor_engine_cpu(); 
    
    namespace tensor{        
        // allocate space for given tensor
        void alloc_space( CTensor1D &ts );
        void alloc_space( CTensor2D &ts );
        void alloc_space( CTensor3D &ts );
        void alloc_space( CTensor4D &ts );
        
        // free space for given tensor
        void free_space( CTensor1D &ts );
        void free_space( CTensor2D &ts );
        void free_space( CTensor3D &ts );
        void free_space( CTensor4D &ts );
        
        // fill the tensor with real value
        void fill( CTensor1D &ts, TENSOR_FLOAT val );
        void fill( CTensor2D &ts, TENSOR_FLOAT val );
        void fill( CTensor3D &ts, TENSOR_FLOAT val );
        void fill( CTensor4D &ts, TENSOR_FLOAT val );
        
        // save tensor to file
        void save_to_file( const CTensor1D &ts, FILE *dst );
        void save_to_file( const CTensor2D &ts, FILE *dst );
        void save_to_file( const CTensor3D &ts, FILE *dst );
        void save_to_file( const CTensor4D &ts, FILE *dst );      
        
        // load tensor from file 
        void load_from_file( CTensor1D &ts, FILE *src );
        void load_from_file( CTensor2D &ts, FILE *src );
        void load_from_file( CTensor3D &ts, FILE *src );
        void load_from_file( CTensor4D &ts, FILE *src );      
        
        // copy data from another tensor
        void copy( CTensor1D &dst, const CTensor1D &src );
        void copy( CTensor2D &dst, const CTensor2D &src );
        void copy( CTensor3D &dst, const CTensor3D &src );
        void copy( CTensor4D &dst, const CTensor4D &src );
    };    
    
    //mapping functions 
    namespace tensor{
        void sigmoid( CTensor1D &mean, const CTensor1D &energy );
        void sigmoid( CTensor2D &mean, const CTensor2D &energy );
        void sigmoid( CTensor3D &mean, const CTensor3D &energy );
        void sigmoid( CTensor4D &mean, const CTensor4D &energy );
    };

    // sampling functions 
    namespace tensor{
        // sample binary distribution
        void sample_binary  ( CTensor1D &state, const CTensor1D &prob );
        void sample_binary  ( CTensor2D &state, const CTensor2D &prob );
        void sample_binary  ( CTensor3D &state, const CTensor3D &prob );
        void sample_binary  ( CTensor4D &state, const CTensor4D &prob );
        
        // sample gaussian distribution with certain sd
        void sample_gaussian( CTensor1D &state, const CTensor1D &mean, TENSOR_FLOAT sd );
        void sample_gaussian( CTensor2D &state, const CTensor2D &mean, TENSOR_FLOAT sd );
        void sample_gaussian( CTensor3D &state, const CTensor3D &mean, TENSOR_FLOAT sd );
        void sample_gaussian( CTensor4D &state, const CTensor4D &mean, TENSOR_FLOAT sd );
        
        // sample gaussian distribution with certain mean sd
        void sample_gaussian( CTensor1D &state, TENSOR_FLOAT sd );        
        void sample_gaussian( CTensor2D &state, TENSOR_FLOAT sd ); 
        void sample_gaussian( CTensor3D &state, TENSOR_FLOAT sd );        
        void sample_gaussian( CTensor4D &state, TENSOR_FLOAT sd );                
    };

    // arithmetic operations
    namespace tensor{
        // dst = a + b
        void add      ( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b );
        void add      ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );
        void add      ( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b );
        void add      ( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b );                
        // dst = a * b, elementwise
        void mul      ( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b );
        void mul      ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );
        void mul      ( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b );
        void mul      ( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b );                
        // dst = a*sa + b*sb
        void scale_add( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        void scale_add( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        void scale_add( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        void scale_add( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );                
        // dst = a - b
        void sub      ( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b );
        void sub      ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );
        void sub      ( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b );
        void sub      ( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b );
        // dst = a + val
        void add      ( CTensor1D &dst, const CTensor1D &a, TENSOR_FLOAT val );
        void add      ( CTensor2D &dst, const CTensor2D &a, TENSOR_FLOAT val );
        void add      ( CTensor3D &dst, const CTensor3D &a, TENSOR_FLOAT val );
        void add      ( CTensor4D &dst, const CTensor4D &a, TENSOR_FLOAT val );
        // dst = a * val
        void mul      ( CTensor1D &dst, const CTensor1D &a, TENSOR_FLOAT val );
        void mul      ( CTensor2D &dst, const CTensor2D &a, TENSOR_FLOAT val );
        void mul      ( CTensor3D &dst, const CTensor3D &a, TENSOR_FLOAT val );
        void mul      ( CTensor4D &dst, const CTensor4D &a, TENSOR_FLOAT val );

        // matrix multiplication
        // dst   = dot( a, b  ) 
        // note: the 1D tensor is treated as 1 * n matrix 
        void dot        ( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b );    
        void dot        ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );            
        // dst  += dot( a, b  ) 
        void sadd__dot  ( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b );    
        void sadd__dot  ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );                    
        // dst  = dot( a   ,  b.T )
        void dot_rt     ( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b );    
        void dot_rt     ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );    
        // dst += dot( a, b.T )
        void sadd__dot_rt( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b );    
        void sadd__dot_rt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );    
        // dst  = dot( a.T , b )
        void dot_lt      ( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b );    
        void sadd__dot_lt( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b );    
        void ssub__dot_lt( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b );    
    };

    // support for error esimtation
    namespace tensor{
        void sadd__abs_err( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b );
        void sadd__abs_err( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );
        void sadd__abs_err( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b );
        void sadd__abs_err( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b );

        void sadd__abs_err_rel( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b );
        void sadd__abs_err_rel( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );
        void sadd__abs_err_rel( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b );
        void sadd__abs_err_rel( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b );
    };
    // support for convolutional RBM
    namespace tensor{
        namespace crbm{
            // normalize by maxpooling 2D
            void norm_maxpooling_2D( CTensor3D &mean, const CTensor3D &energy, int pool_size );

            // sample the data using 2D maxpooling 
            void sample_maxpooling_2D( CTensor3D &state, const CTensor3D &mean, int pool_size );

            // fit the last two dimension of src into dst's size, copy the fitted part into dst
            void copy_fit( CTensor2D &dst, const CTensor2D &src );
            void copy_fit( CTensor3D &dst, const CTensor3D &src );
            
            // pool up
            void pool_up( CTensor3D &dst , const CTensor3D &src, int pool_size ); 
            
            // 2D convolution with bias
            // convolution, leaves the valid area
            // dst = (~a) (*)  filter + bias 
            void conv2_r_valid     ( CTensor3D &dst, const CTensor3D &a, const CTensor4D &filter, const CTensor1D &bias );
            
            // dst = ( a) (*) filter + bias
            void conv2_full        ( CTensor3D &dst, const CTensor3D &a, const CTensor4D &filter, const CTensor1D &bias );
            
            // convolution with big filter
            void sadd__conv2_r_big_filter( CTensor4D &dst, const CTensor3D &a, const CTensor3D &b );
            void ssub__conv2_r_big_filter( CTensor4D &dst, const CTensor3D &a, const CTensor3D &b );
            
            // sum over last two dimension
            void sadd__sum_2D( CTensor1D &dst, const CTensor3D &src );
            void ssub__sum_2D( CTensor1D &dst, const CTensor3D &src );
            
            // calculate information of sparse regularization
            void add_sparse_info( CTensor1D &sum_mf, CTensor1D &sum_mf_grad, const CTensor3D &src, int pool_size );
        };        
    };

    // host only code
    namespace cpu_only{
        // average value
        TENSOR_FLOAT sum( const CTensor1D &a );
        TENSOR_FLOAT sum( const CTensor2D &a );
        TENSOR_FLOAT sum( const CTensor3D &a );
        TENSOR_FLOAT sum( const CTensor4D &a );
        
        // average value
        TENSOR_FLOAT avg( const CTensor1D &a );
        TENSOR_FLOAT avg( const CTensor2D &a );
        TENSOR_FLOAT avg( const CTensor3D &a );
        TENSOR_FLOAT avg( const CTensor4D &a );

        // variance
        TENSOR_FLOAT var( const CTensor1D &a );
        TENSOR_FLOAT var( const CTensor2D &a );
        TENSOR_FLOAT var( const CTensor3D &a );
        TENSOR_FLOAT var( const CTensor4D &a );
        
        // standard variance
        TENSOR_FLOAT std_var( const CTensor1D &a );
        TENSOR_FLOAT std_var( const CTensor2D &a );
        TENSOR_FLOAT std_var( const CTensor3D &a );
        TENSOR_FLOAT std_var( const CTensor4D &a );
        
        // min value
        TENSOR_FLOAT min_value( const CTensor1D &a );
        TENSOR_FLOAT min_value( const CTensor2D &a );
        TENSOR_FLOAT min_value( const CTensor3D &a );
        TENSOR_FLOAT min_value( const CTensor4D &a );
        // max value
        TENSOR_FLOAT max_value( const CTensor1D &a );
        TENSOR_FLOAT max_value( const CTensor2D &a );
        TENSOR_FLOAT max_value( const CTensor3D &a );
        TENSOR_FLOAT max_value( const CTensor4D &a );
    };
};
#endif


