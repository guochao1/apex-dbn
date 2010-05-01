#ifndef _APEX_TENSOR_GPU_H_
#define _APEX_TENSOR_GPU_H_

#include "apex_op_plan.h"
#include "apex_tensor.h"

// data structure for tensor
namespace apex_tensor{
    struct GTensor1D{
        int           x_max;        
        unsigned int  pitch;
        TENSOR_FLOAT *elem;
        // stream dependecy, this variable is private
        int           __stream_dep;
        GTensor1D(){ __stream_dep = 0; }
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
        inline GTensor1D& operator += ( const apex_op_plan::Sum2DPlan      <GTensor3D> &val );        
        inline GTensor1D& operator -= ( const apex_op_plan::Sum2DPlan      <GTensor3D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::ClonePlan      <CTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::AllocLikePlan  <CTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::AllocLikePlan  <GTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::SigmoidPlan      <GTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::SampleBinaryPlan <GTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::SampleGaussianPlan<GTensor1D,TENSOR_FLOAT> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::AddPlan <GTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::SubPlan <GTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::MulPlan <GTensor1D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::DotPlan  <GTensor1D,GTensor2D> &val );        
        inline GTensor1D& operator += ( const apex_op_plan::DotPlan  <GTensor1D,GTensor2D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::DotRTPlan<GTensor1D,GTensor2D> &val );        
        inline GTensor1D& operator += ( const apex_op_plan::DotRTPlan<GTensor1D,GTensor2D> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::ScalePlan<GTensor1D,TENSOR_FLOAT> &val );        
        inline GTensor1D& operator += ( const apex_op_plan::ScalePlan<GTensor1D,TENSOR_FLOAT> &val );        
        inline GTensor1D& operator -= ( const apex_op_plan::ScalePlan<GTensor1D,TENSOR_FLOAT> &val );        
        inline GTensor1D& operator =  ( const apex_op_plan::ScaleAddPlan<GTensor1D,TENSOR_FLOAT> &val );        
        inline GTensor1D& operator += ( const apex_op_plan::ScaleAddPlan<GTensor1D,TENSOR_FLOAT> &val );       
        inline GTensor1D& operator -= ( const apex_op_plan::ScaleAddPlan<GTensor1D,TENSOR_FLOAT> &val );        
    };

    struct GTensor2D{
        int           x_max, y_max;        
        unsigned int  pitch;
        TENSOR_FLOAT *elem;
        // stream dependecy, this variable is private
        int           __stream_dep;
        GTensor2D(){ __stream_dep = 0; }       
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
        inline GTensor2D& operator =  ( const apex_op_plan::AllocLikePlan  <GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::SigmoidPlan        <GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::SampleBinaryPlan   <GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::SampleGaussianPlan <GTensor2D,TENSOR_FLOAT> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::AddPlan <GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::SubPlan <GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::MulPlan <GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::DotPlan  <GTensor2D,GTensor2D> &val );        
        inline GTensor2D& operator += ( const apex_op_plan::DotPlan  <GTensor2D,GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::DotRTPlan<GTensor2D,GTensor2D> &val );        
        inline GTensor2D& operator += ( const apex_op_plan::DotRTPlan<GTensor2D,GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::DotLTPlan<GTensor1D,GTensor1D> &val );        
        inline GTensor2D& operator += ( const apex_op_plan::DotLTPlan<GTensor1D,GTensor1D> &val );        
        inline GTensor2D& operator -= ( const apex_op_plan::DotLTPlan<GTensor1D,GTensor1D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::ScalePlan<GTensor2D,TENSOR_FLOAT> &val );        
        inline GTensor2D& operator += ( const apex_op_plan::ScalePlan<GTensor2D,TENSOR_FLOAT> &val );        
        inline GTensor2D& operator -= ( const apex_op_plan::ScalePlan<GTensor2D,TENSOR_FLOAT> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::ScaleAddPlan<GTensor2D,TENSOR_FLOAT> &val );        
        inline GTensor2D& operator += ( const apex_op_plan::ScaleAddPlan<GTensor2D,TENSOR_FLOAT> &val );        
        inline GTensor2D& operator -= ( const apex_op_plan::ScaleAddPlan<GTensor2D,TENSOR_FLOAT> &val );        
        // support for sparse operation
        inline GTensor2D& operator =  ( const apex_op_plan::DotPlan  <GTensor2DSparse,GTensor2D> &val );        
        inline GTensor2D& operator += ( const apex_op_plan::DotPlan  <GTensor2DSparse,GTensor2D> &val );        
        inline GTensor2D& operator =  ( const apex_op_plan::DotLTPlan<GTensor2DSparse,GTensor2D> &val );        
        inline GTensor2D& operator += ( const apex_op_plan::DotLTPlan<GTensor2DSparse,GTensor2D> &val );        
        inline GTensor2D& operator -= ( const apex_op_plan::DotLTPlan<GTensor2DSparse,GTensor2D> &val );        
    };

    struct GTensor3D{
        int           x_max, y_max, z_max;                
        unsigned int  pitch;
        TENSOR_FLOAT *elem;
        // stream dependecy, this variable is private
        int           __stream_dep;
        GTensor3D(){ __stream_dep = 0; }
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
        inline GTensor3D& operator =  ( const apex_op_plan::ClonePlan      <CTensor3D> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::AllocLikePlan  <CTensor3D> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::AllocLikePlan  <GTensor3D> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::SigmoidPlan     <GTensor3D> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::SampleBinaryPlan  <GTensor3D> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::SampleGaussianPlan<GTensor3D,TENSOR_FLOAT> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::AddPlan<GTensor3D> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::SubPlan<GTensor3D> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::MulPlan<GTensor3D> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::ScalePlan<GTensor3D,TENSOR_FLOAT> &val );        
        inline GTensor3D& operator += ( const apex_op_plan::ScalePlan<GTensor3D,TENSOR_FLOAT> &val );        
        inline GTensor3D& operator -= ( const apex_op_plan::ScalePlan<GTensor3D,TENSOR_FLOAT> &val );        
        inline GTensor3D& operator =  ( const apex_op_plan::ScaleAddPlan<GTensor3D,TENSOR_FLOAT> &val );       
        inline GTensor3D& operator += ( const apex_op_plan::ScaleAddPlan<GTensor3D,TENSOR_FLOAT> &val );        
        inline GTensor3D& operator -= ( const apex_op_plan::ScaleAddPlan<GTensor3D,TENSOR_FLOAT> &val );        
    };
    
    struct GTensor4D{
        int           x_max, y_max, z_max, h_max;        
        unsigned int  pitch;
        TENSOR_FLOAT *elem;
        // stream dependecy, this variable is private
        int           __stream_dep;
        GTensor4D(){ __stream_dep = 0; }
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
        inline GTensor4D& operator =  ( const apex_op_plan::ClonePlan      <CTensor4D> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::AllocLikePlan  <CTensor4D> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::AllocLikePlan  <GTensor4D> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::SigmoidPlan<GTensor4D> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::SampleBinaryPlan  <GTensor4D> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::SampleGaussianPlan<GTensor4D,TENSOR_FLOAT> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::AddPlan<GTensor4D> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::SubPlan<GTensor4D> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::MulPlan<GTensor4D> &val );        
        inline GTensor4D& operator =  ( const apex_op_plan::ScalePlan<GTensor4D,TENSOR_FLOAT> &val );                        
        inline GTensor4D& operator += ( const apex_op_plan::ScalePlan<GTensor4D,TENSOR_FLOAT> &val );                        
        inline GTensor4D& operator -= ( const apex_op_plan::ScalePlan<GTensor4D,TENSOR_FLOAT> &val );                        
        inline GTensor4D& operator =  ( const apex_op_plan::ScaleAddPlan<GTensor4D,TENSOR_FLOAT> &val );        
        inline GTensor4D& operator += ( const apex_op_plan::ScaleAddPlan<GTensor4D,TENSOR_FLOAT> &val );        
        inline GTensor4D& operator -= ( const apex_op_plan::ScaleAddPlan<GTensor4D,TENSOR_FLOAT> &val );        
    };
    
    /* sparse tensor support */
    /** 
        index structure for sparse tensor
    */
    struct GSparseIndex2D{
        /** index for y and x dimension */
        int *y, *x;
        /** length of current index */
        unsigned int length;
        /** allocated length of current index, maximum number of element supported*/
        unsigned int alloc_length;
        GSparseIndex2D(){}        
        inline GSparseIndex2D & operator = ( const apex_op_plan::ClonePlan<CSparseIndex2D> &val );
    };
    /** 
        sparse 2D tensor
     */
    struct GTensor2DSparse{
        GSparseIndex2D index;
        TENSOR_FLOAT  *elem;
        GTensor2DSparse(){}

        inline GTensor2DSparse & operator+= ( const GTensor2DSparse & b ); 
        inline GTensor2DSparse & operator-= ( const GTensor2DSparse & b ); 
        inline apex_op_plan::TransposePlan<GTensor2DSparse> T() const;
        inline GTensor2DSparse & operator = ( const apex_op_plan::SubPlan<GTensor2DSparse> &val );      
        inline GTensor2DSparse & operator = ( const apex_op_plan::DotRTPlan  <GTensor2D,GTensor2D> &val );        
        inline GTensor2DSparse & operator+= ( const apex_op_plan::DotRTPlan  <GTensor2D,GTensor2D> &val );        
    };

    // functions related to sparse tensor 
    namespace tensor{
        // allocate space for index 
        void alloc_space_index( GSparseIndex2D &index ); 
        // allocate space using setting of index
        GTensor2DSparse alloc_space_data( GSparseIndex2D index );                        
        // free the index space 
        void free_space_index( GSparseIndex2D  &index );
        // free data space of tensor
        void free_space_data ( GTensor2DSparse &ts );
        // copy index from cpu to gpu
        void copy_index ( GSparseIndex2D &dst , const CSparseIndex2D &a  );        
        // copy from cpu to gpu
        void copy_data  ( GTensor2DSparse &dst, const CTensor2DSparse &a ); 
        // copy from gpu to cpu
        void copy_data  ( CTensor2DSparse &dst, const GTensor2DSparse &a ); 
    };

    namespace tensor{
        // dst = a + b;
        void add   ( GTensor2DSparse &dst , const GTensor2DSparse &a, const GTensor2DSparse &b );
        // dst = a - b;
        void sub   ( GTensor2DSparse &dst , const GTensor2DSparse &a, const GTensor2DSparse &b );        
        // dst = dot( a, b.T );
        void dot_rt      ( GTensor2DSparse &dst , const GTensor2D &a      , const GTensor2D &b );
        void sadd__dot_rt( GTensor2DSparse &dst , const GTensor2D &a      , const GTensor2D &b );
        // dst = dot( W, P )
        void dot       ( GTensor2D &dst , const GTensor2DSparse &W, const GTensor2D &P );
        void sadd__dot ( GTensor2D &dst , const GTensor2DSparse &W, const GTensor2D &P );
        // dst = dot( W.T,P )
        void dot_lt      ( GTensor2D &dst , const GTensor2DSparse &W, const GTensor2D &P );        
        void sadd__dot_lt( GTensor2D &dst , const GTensor2DSparse &W, const GTensor2D &P );        
        void ssub__dot_lt( GTensor2D &dst , const GTensor2DSparse &W, const GTensor2D &P );        
    };
    /*-------------------------------------*/

    
    // functions defined for tensor
    // intialize the tensor engine for use, seed is 
    // the seed for random number generator
    void init_tensor_engine_gpu( int seed );
    // this function is called when the program exits
    void destroy_tensor_engine_gpu();
    // initialize the asynchronize stream engine
    void init_stream_engine_gpu( int num_stream );
    // destroy asynchronize stream engine
    void destroy_stream_engine_gpu();

    // sync gpu threads , wait until all gpu operations complete, 
    // this functions is used for timing
    void sync_gpu_threads();
    
    // support for asynchronize execution
    namespace async{
        // set the dependecy of a data to stream,
        // the setting operation on the data will be asynchronized, by default stream_id = 0
        // if the stream id is invalid, then the stream will be set to default value
        void set_dependecy( GTensor1D &ts, int stream_id );
        void set_dependecy( GTensor2D &ts, int stream_id );
        void set_dependecy( GTensor3D &ts, int stream_id );
        void set_dependecy( GTensor4D &ts, int stream_id );
    };

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
        void copy( CTensor1D &dst, const GTensor1D &src );
        void copy( CTensor2D &dst, const GTensor2D &src );
        void copy( CTensor3D &dst, const GTensor3D &src );
        void copy( CTensor4D &dst, const GTensor4D &src );

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
        
        // dst += a*sa + b*sb
        void sadd__scale_add( GTensor1D &dst, const GTensor1D &a, const GTensor1D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        void sadd__scale_add( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        void sadd__scale_add( GTensor3D &dst, const GTensor3D &a, const GTensor3D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        void sadd__scale_add( GTensor4D &dst, const GTensor4D &a, const GTensor4D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );                

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

        // dst += a * val
        void sadd__mul      ( GTensor1D &dst, const GTensor1D &a, TENSOR_FLOAT val );
        void sadd__mul      ( GTensor2D &dst, const GTensor2D &a, TENSOR_FLOAT val );
        void sadd__mul      ( GTensor3D &dst, const GTensor3D &a, TENSOR_FLOAT val );
        void sadd__mul      ( GTensor4D &dst, const GTensor4D &a, TENSOR_FLOAT val );

        // matrix multiplication
        // dst   = dot( a, b  ) 
        // note: the 1D tensor is treated as 1 * n matrix 
        void dot      ( GTensor1D &dst, const GTensor1D &a, const GTensor2D &b );    
        void dot      ( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );            
        // dst  += dot( a, b  ) 
        void sadd__dot  ( GTensor1D &dst, const GTensor1D &a, const GTensor2D &b );    
        void sadd__dot  ( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );                    
        // dst  = dot( a   ,  b.T )
        void dot_rt    ( GTensor1D &dst, const GTensor1D &a, const GTensor2D &b );    
        void dot_rt    ( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );    
        // dst += dot( a, b.T )
        void sadd__dot_rt( GTensor1D &dst, const GTensor1D &a, const GTensor2D &b );    
        void sadd__dot_rt( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );    
        // dst  = dot( a.T , b )
        void dot_lt      ( GTensor2D &dst, const GTensor1D &a, const GTensor1D &b );    
        void sadd__dot_lt( GTensor2D &dst, const GTensor1D &a, const GTensor1D &b );    
        void ssub__dot_lt( GTensor2D &dst, const GTensor1D &a, const GTensor1D &b );    
    };

    // support for error esimtate
    namespace tensor{
        void sadd__abs_err( GTensor1D &dst, const GTensor1D &a, const GTensor1D &b );
        void sadd__abs_err( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );
        void sadd__abs_err( GTensor3D &dst, const GTensor3D &a, const GTensor3D &b );
        void sadd__abs_err( GTensor4D &dst, const GTensor4D &a, const GTensor4D &b );

        void sadd__abs_err_rel( GTensor1D &dst, const GTensor1D &a, const GTensor1D &b );
        void sadd__abs_err_rel( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );
        void sadd__abs_err_rel( GTensor3D &dst, const GTensor3D &a, const GTensor3D &b );
        void sadd__abs_err_rel( GTensor4D &dst, const GTensor4D &a, const GTensor4D &b );
        // ignore x < 1e-5
        void sadd__abs_err_relT( GTensor1D &dst, const GTensor1D &a, const GTensor1D &b );
        void sadd__abs_err_relT( GTensor2D &dst, const GTensor2D &a, const GTensor2D &b );
        void sadd__abs_err_relT( GTensor3D &dst, const GTensor3D &a, const GTensor3D &b );
        void sadd__abs_err_relT( GTensor4D &dst, const GTensor4D &a, const GTensor4D &b );
    }
    
    // support for convolutional RBM
    namespace tensor{
        namespace crbm{
            // fit the last two dimension of src into dst's size, copy the fitted part into dst
            void copy_fit( GTensor2D &dst, const CTensor2D &src );
            void copy_fit( GTensor3D &dst, const CTensor3D &src );
            void copy_fit( GTensor3D &dst, const GTensor3D &src );
            
            // fill the edge of dest by values in src
            void refill_edge_area( GTensor3D &dst, const GTensor3D &src, int edge_y_len, int edge_x_len );

            // normalize by maxpooling 2D
            void norm_maxpooling_2D( GTensor3D &mean, const GTensor3D &energy, int pool_size );

            // sample the data using 2D maxpooling 
            void sample_maxpooling_2D( GTensor3D &state, const GTensor3D &mean, int pool_size );
            
            // pool up
            void pool_up( GTensor3D &dst , const GTensor3D &src, int pool_size );             
            
            // 2D convolution with bias
            // convolution, leaves the valid area
            // dst = (~a) (*)  filter + bias 
            void conv2_r_valid     ( GTensor3D &dst, const GTensor3D &a, const GTensor4D &filter, const GTensor1D &bias );
            
            // dst = ( a) (*) filter + bias
            void conv2_full        ( GTensor3D &dst, const GTensor3D &a, const GTensor4D &filter, const GTensor1D &bias );
            
            // convolution with big filter
            void sadd__conv2_r_big_filter( GTensor4D &dst, const GTensor3D &a, const GTensor3D &b );
            void ssub__conv2_r_big_filter( GTensor4D &dst, const GTensor3D &a, const GTensor3D &b );
            
            // sum over last two dimension
            void sadd__sum_2D( GTensor1D &dst, const GTensor3D &src );
            void ssub__sum_2D( GTensor1D &dst, const GTensor3D &src );

            // add last two dimension 
            void sum_2D    ( GTensor2D &dst, const GTensor4D &src );            
            void sadd__scale( GTensor4D &dst, const GTensor2D &src, TENSOR_FLOAT scale_src );
            
            // calculate information of sparse regularization
            void add_sparse_info( GTensor1D &sum_mf, GTensor1D &sum_mf_grad, const GTensor3D &src, int pool_size );
        };        
    };
};

#endif


