#ifndef _APEX_TENSOR_CPU_H_
#define _APEX_TENSOR_CPU_H_

#include "apex_op_plan.h"
#include "apex_tensor.h"
#include <vector>

// data structure for tensor
namespace apex_tensor{
    struct CTensor1D{
        int           x_max;        
        unsigned int  pitch;
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
        inline CTensor1D& operator += ( const apex_op_plan::Sum2DPlan         <CTensor3D> &val );        
        inline CTensor1D& operator -= ( const apex_op_plan::Sum2DPlan         <CTensor3D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::ClonePlan         <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::AllocLikePlan     <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::AllocLikePlan     <GTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::SigmoidPlan       <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::SampleBinaryPlan  <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::SampleGaussianPlan<CTensor1D,TENSOR_FLOAT> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::AddPlan  <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::SubPlan  <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::MulPlan  <CTensor1D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::DotPlan  <CTensor1DSparse,CTensor2D> &val );  
        inline CTensor1D& operator += ( const apex_op_plan::DotPlan  <CTensor1DSparse,CTensor2D> &val );        
        inline CTensor1D& operator -= ( const apex_op_plan::DotPlan  <CTensor1DSparse,CTensor2D> &val );        
        inline CTensor1D& operator += ( const apex_op_plan::ScalePlan<apex_op_plan::DotPlan<CTensor1DSparse,CTensor2D>,TENSOR_FLOAT> &val );        
        inline CTensor1D& operator -= ( const apex_op_plan::ScalePlan<apex_op_plan::DotPlan<CTensor1DSparse,CTensor2D>,TENSOR_FLOAT> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::DotPlan  <CTensor1D,CTensor2D> &val );        
        inline CTensor1D& operator += ( const apex_op_plan::DotPlan  <CTensor1D,CTensor2D> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::DotRTPlan<CTensor1D,CTensor2D> &val );        
        inline CTensor1D& operator += ( const apex_op_plan::DotRTPlan<CTensor1D,CTensor2D> &val );        
        inline CTensor1D& operator += ( const apex_op_plan::ScalePlan<CTensor1DSparse,TENSOR_FLOAT> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::ScalePlan<CTensor1D,TENSOR_FLOAT> &val );        
        inline CTensor1D& operator += ( const apex_op_plan::ScalePlan<CTensor1D,TENSOR_FLOAT> &val );        
        inline CTensor1D& operator -= ( const apex_op_plan::ScalePlan<CTensor1D,TENSOR_FLOAT> &val );        
        inline CTensor1D& operator =  ( const apex_op_plan::ScaleAddPlan<CTensor1D,TENSOR_FLOAT> &val );        
        inline CTensor1D& operator += ( const apex_op_plan::ScaleAddPlan<CTensor1D,TENSOR_FLOAT> &val );        
        inline CTensor1D& operator -= ( const apex_op_plan::ScaleAddPlan<CTensor1D,TENSOR_FLOAT> &val );        
    };

    struct CTensor2D{
        int           x_max, y_max;        
        unsigned int  pitch;
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
        inline CTensor2D& operator =  ( const apex_op_plan::SampleGaussianPlan<CTensor2D,TENSOR_FLOAT> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::AddPlan  <CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::SubPlan  <CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::MulPlan  <CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::DotPlan  <CTensor2D,CTensor2D> &val );        
        inline CTensor2D& operator += ( const apex_op_plan::DotPlan  <CTensor2D,CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::DotRTPlan<CTensor2D,CTensor2D> &val );        
        inline CTensor2D& operator += ( const apex_op_plan::DotRTPlan<CTensor2D,CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::DotLTPlan<CTensor1DSparse,CTensor1D> &val );  
        inline CTensor2D& operator += ( const apex_op_plan::DotLTPlan<CTensor1DSparse,CTensor1D> &val );        
        inline CTensor2D& operator -= ( const apex_op_plan::DotLTPlan<CTensor1DSparse,CTensor1D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::DotLTPlan<CTensor1D,CTensor1D> &val );        
        inline CTensor2D& operator += ( const apex_op_plan::DotLTPlan<CTensor1D,CTensor1D> &val );        
        inline CTensor2D& operator -= ( const apex_op_plan::DotLTPlan<CTensor1D,CTensor1D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::ScalePlan<CTensor2D,TENSOR_FLOAT> &val );        
        inline CTensor2D& operator += ( const apex_op_plan::ScalePlan<CTensor2D,TENSOR_FLOAT> &val );        
        inline CTensor2D& operator -= ( const apex_op_plan::ScalePlan<CTensor2D,TENSOR_FLOAT> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::ScaleAddPlan<CTensor2D,TENSOR_FLOAT> &val );        
        inline CTensor2D& operator += ( const apex_op_plan::ScaleAddPlan<CTensor2D,TENSOR_FLOAT> &val );        
        inline CTensor2D& operator -= ( const apex_op_plan::ScaleAddPlan<CTensor2D,TENSOR_FLOAT> &val );  

        inline CTensor2D& operator+= ( const apex_op_plan::ScalePlan<apex_op_plan::DotLTPlan<CTensor1DSparse,CTensor1D>,TENSOR_FLOAT> &val );
        // support for sparse operation
        inline CTensor2D& operator =  ( const apex_op_plan::DotPlan  <CTensor2DSparse,CTensor2D> &val );        
        inline CTensor2D& operator += ( const apex_op_plan::DotPlan  <CTensor2DSparse,CTensor2D> &val );        
        inline CTensor2D& operator =  ( const apex_op_plan::DotLTPlan<CTensor2DSparse,CTensor2D> &val );        
        inline CTensor2D& operator += ( const apex_op_plan::DotLTPlan<CTensor2DSparse,CTensor2D> &val );        
        inline CTensor2D& operator -= ( const apex_op_plan::DotLTPlan<CTensor2DSparse,CTensor2D> &val );        
    };

    struct CTensor3D{
        int           x_max, y_max, z_max;                
        unsigned int  pitch;
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
        inline CTensor3D& operator =  ( const apex_op_plan::SampleGaussianPlan<CTensor3D,TENSOR_FLOAT> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::AddPlan<CTensor3D> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::SubPlan<CTensor3D> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::MulPlan<CTensor3D> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::ScalePlan<CTensor3D,TENSOR_FLOAT> &val );        
        inline CTensor3D& operator += ( const apex_op_plan::ScalePlan<CTensor3D,TENSOR_FLOAT> &val );        
        inline CTensor3D& operator -= ( const apex_op_plan::ScalePlan<CTensor3D,TENSOR_FLOAT> &val );        
        inline CTensor3D& operator =  ( const apex_op_plan::ScaleAddPlan<CTensor3D,TENSOR_FLOAT> &val );        
        inline CTensor3D& operator += ( const apex_op_plan::ScaleAddPlan<CTensor3D,TENSOR_FLOAT> &val );        
        inline CTensor3D& operator -= ( const apex_op_plan::ScaleAddPlan<CTensor3D,TENSOR_FLOAT> &val );        
    };
    
    struct CTensor4D{
        int           x_max, y_max, z_max, h_max;        
        unsigned int  pitch;

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
        inline CTensor4D& operator =  ( const apex_op_plan::SampleGaussianPlan<CTensor4D,TENSOR_FLOAT> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::AddPlan<CTensor4D> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::SubPlan<CTensor4D> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::MulPlan<CTensor4D> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::ScalePlan<CTensor4D,TENSOR_FLOAT> &val );        
        inline CTensor4D& operator += ( const apex_op_plan::ScalePlan<CTensor4D,TENSOR_FLOAT> &val );        
        inline CTensor4D& operator -= ( const apex_op_plan::ScalePlan<CTensor4D,TENSOR_FLOAT> &val );        
        inline CTensor4D& operator =  ( const apex_op_plan::ScaleAddPlan<CTensor4D,TENSOR_FLOAT> &val );        
        inline CTensor4D& operator += ( const apex_op_plan::ScaleAddPlan<CTensor4D,TENSOR_FLOAT> &val );        
        inline CTensor4D& operator -= ( const apex_op_plan::ScaleAddPlan<CTensor4D,TENSOR_FLOAT> &val );        
    };

    /* sparse tensor support */
    /** 
        index structure for sparse tensor
    */
    struct CSparseIndex1D{
        /** index for y and x dimension */
        int *x;
        /** length of current index */
        unsigned int length;
        /** allocated length of current index, maximum number of element supported*/
        unsigned int alloc_length;
        CSparseIndex1D(){}     
        inline bool operator==( const CSparseIndex1D &b ) const{
            return x == b.x && length == b.length; 
        }
    };
    struct CSparseIndex2D{
        /** index for y and x dimension */
        int *y, *x;
        /** length of current index */
        unsigned int length;
        /** allocated length of current index, maximum number of element supported*/
        unsigned int alloc_length;
        CSparseIndex2D(){}     
        inline bool operator==( const CSparseIndex2D &b ) const{
            return y==b.y && x == b.x && length == b.length; 
        }
        inline CSparseIndex2D & operator = ( const apex_op_plan::ClonePlan<CSparseIndex2D> &val );
    };
    /** 
        sparse tensor
     */
    struct CTensor1DSparse{
       CSparseIndex1D  index;
        TENSOR_FLOAT   *elem;        
        CTensor1DSparse(){}

        inline apex_op_plan::TransposePlan<CTensor1DSparse> T() const;
    };

    struct CTensor2DSparse{
        CSparseIndex2D index;
        TENSOR_FLOAT  *elem;
        CTensor2DSparse(){}

        inline CTensor2DSparse & operator+= ( const CTensor2DSparse & b ); 
        inline CTensor2DSparse & operator-= ( const CTensor2DSparse & b ); 
        inline apex_op_plan::TransposePlan<CTensor2DSparse> T() const;
        inline CTensor2DSparse & operator = ( const apex_op_plan::SubPlan<CTensor2DSparse> &val );      
        inline CTensor2DSparse & operator = ( const apex_op_plan::DotRTPlan  <CTensor2D,CTensor2D> &val );      
        inline CTensor2DSparse & operator+= ( const apex_op_plan::DotRTPlan  <CTensor2D,CTensor2D> &val );      
    };

    namespace tensor{
        // allocate space for index 
        inline void alloc_space_index( CSparseIndex2D &index ); 
        // allocate space using setting of index
        inline CTensor2DSparse alloc_space_data( CSparseIndex2D index );                 
        // free the index space 
        inline void free_space_index( CSparseIndex2D  &index );
        // free data space of tensor
        inline void free_space_data ( CTensor2DSparse &ts );
        // copy index from cpu to cpu
        inline void copy_index ( CSparseIndex2D &dst , const CSparseIndex2D &a );
        // copy from cpu to cpu
        inline void copy_data  ( CTensor2DSparse &dst, const CTensor2DSparse &a );                
    };

    // functions related to sparse tensor 
    namespace tensor{
        // free space 
        inline void free_space( CTensor1DSparse &sps );
        // create sparse matrix by vector
        inline CTensor1DSparse create_sparse( const std::vector<int> &idx, const std::vector<TENSOR_FLOAT> &vals );
        // dst += a * scale
        inline void sadd__mul( CTensor1D &dst , const CTensor1DSparse &a, TENSOR_FLOAT val );
        // dst = dot( a   , b );
        inline void dot      ( CTensor1D &dst, const CTensor1DSparse &a, const CTensor2D &b );
        inline void sadd__dot( CTensor1D &dst, const CTensor1DSparse &a, const CTensor2D &b );
        inline void ssub__dot( CTensor1D &dst, const CTensor1DSparse &a, const CTensor2D &b );
        // dst-= scale * dot( a, b );
        inline void sadd__dot_scale( CTensor1D &dst, const CTensor1DSparse &a, const CTensor2D &b, TENSOR_FLOAT scale );
        inline void ssub__dot_scale( CTensor1D &dst, const CTensor1DSparse &a, const CTensor2D &b, TENSOR_FLOAT scale );
        // dst = dot( a.T , b );
        inline void dot_lt      ( CTensor2D &dst, const CTensor1DSparse &a, const CTensor1D &b );        
        inline void sadd__dot_lt( CTensor2D &dst, const CTensor1DSparse &a, const CTensor1D &b );        
        inline void ssub__dot_lt( CTensor2D &dst, const CTensor1DSparse &a, const CTensor1D &b );        
        // dst += dot( a.T , b ) *scale
        inline void sadd__dot_lt_scale( CTensor2D &dst, const CTensor1DSparse &a, const CTensor1D &b, TENSOR_FLOAT scale );        
        // dst = sum( a * b );
        inline TENSOR_FLOAT sum_mul( const CTensor1DSparse &a, const CTensor1D &b );        
    };

    namespace tensor{
        // dst = a + b;
        inline void add   ( CTensor2DSparse &dst , const CTensor2DSparse &a, const CTensor2DSparse &b );
        // dst = a - b;
        inline void sub   ( CTensor2DSparse &dst , const CTensor2DSparse &a, const CTensor2DSparse &b );        
        
        // dst = dot( a, b.T );
        inline void dot_rt( CTensor2DSparse &dst , const CTensor2D &a , const CTensor2D &b );
        inline void sadd__dot_rt( CTensor2DSparse &dst , const CTensor2D &a, const CTensor2D &b );      
        // dst = dot( W, P )
        inline void dot       ( CTensor2D &dst , const CTensor2DSparse &W, const CTensor2D &P );
        inline void sadd__dot ( CTensor2D &dst , const CTensor2DSparse &W, const CTensor2D &P );        
        // dst = dot( W.T,P )
        inline void dot_lt      ( CTensor2D &dst , const CTensor2DSparse &W, const CTensor2D &P );        
        inline void sadd__dot_lt( CTensor2D &dst , const CTensor2DSparse &W, const CTensor2D &P );        
        inline void ssub__dot_lt( CTensor2D &dst , const CTensor2DSparse &W, const CTensor2D &P );        
    };
    /*-------------------------------------*/

    
    // functions defined for tensor

    // intialize the tensor engine for use
    void init_tensor_engine_cpu( int seed );
    // this function is called when the program exits
    void destroy_tensor_engine_cpu(); 
    
    namespace async{
        inline void set_dependecy( CTensor1D &ts, int stream_id );
        inline void set_dependecy( CTensor2D &ts, int stream_id );
        inline void set_dependecy( CTensor3D &ts, int stream_id );
        inline void set_dependecy( CTensor4D &ts, int stream_id );
    };
    
    namespace tensor{        
        // allocate space for given tensor
        inline void alloc_space( CTensor1D &ts );
        inline void alloc_space( CTensor2D &ts );
        inline void alloc_space( CTensor3D &ts );
        inline void alloc_space( CTensor4D &ts );
        
        // free space for given tensor
        inline void free_space( CTensor1D &ts );
        inline void free_space( CTensor2D &ts );
        inline void free_space( CTensor3D &ts );
        inline void free_space( CTensor4D &ts );
        
        // fill the tensor with real value
        inline void fill( CTensor1D &ts, TENSOR_FLOAT val );
        inline void fill( CTensor2D &ts, TENSOR_FLOAT val );
        inline void fill( CTensor3D &ts, TENSOR_FLOAT val );
        inline void fill( CTensor4D &ts, TENSOR_FLOAT val );
        
        // save tensor to file
        inline void save_to_file( const CTensor1D &ts, FILE *dst );
        inline void save_to_file( const CTensor2D &ts, FILE *dst );
        inline void save_to_file( const CTensor3D &ts, FILE *dst );
        inline void save_to_file( const CTensor4D &ts, FILE *dst );      
        
        // load tensor from file 
        inline void load_from_file( CTensor1D &ts, FILE *src );
        inline void load_from_file( CTensor2D &ts, FILE *src );
        inline void load_from_file( CTensor3D &ts, FILE *src );
        inline void load_from_file( CTensor4D &ts, FILE *src );      
        
        // copy data from another tensor
        inline void copy( CTensor1D &dst, const CTensor1D &src );
        inline void copy( CTensor2D &dst, const CTensor2D &src );
        inline void copy( CTensor3D &dst, const CTensor3D &src );
        inline void copy( CTensor4D &dst, const CTensor4D &src );
    };    
    
    //mapping functions 
    namespace tensor{
        inline void sigmoid( CTensor1D &mean, const CTensor1D &energy );
        inline void sigmoid( CTensor2D &mean, const CTensor2D &energy );
        inline void sigmoid( CTensor3D &mean, const CTensor3D &energy );
        inline void sigmoid( CTensor4D &mean, const CTensor4D &energy );
    };

    // sampling functions 
    namespace tensor{
        // sample binary distribution
        inline void sample_binary  ( CTensor1D &state, const CTensor1D &prob );
        inline void sample_binary  ( CTensor2D &state, const CTensor2D &prob );
        inline void sample_binary  ( CTensor3D &state, const CTensor3D &prob );
        inline void sample_binary  ( CTensor4D &state, const CTensor4D &prob );
        
        // sample gaussian distribution with certain sd
        inline void sample_gaussian( CTensor1D &state, const CTensor1D &mean, TENSOR_FLOAT sd );
        inline void sample_gaussian( CTensor2D &state, const CTensor2D &mean, TENSOR_FLOAT sd );
        inline void sample_gaussian( CTensor3D &state, const CTensor3D &mean, TENSOR_FLOAT sd );
        inline void sample_gaussian( CTensor4D &state, const CTensor4D &mean, TENSOR_FLOAT sd );
        
        // sample gaussian distribution with certain mean sd
        inline void sample_gaussian( CTensor1D &state, TENSOR_FLOAT sd );        
        inline void sample_gaussian( CTensor2D &state, TENSOR_FLOAT sd ); 
        inline void sample_gaussian( CTensor3D &state, TENSOR_FLOAT sd );        
        inline void sample_gaussian( CTensor4D &state, TENSOR_FLOAT sd );                
    };

    // arithmetic operations
    namespace tensor{
        // dst = a + b
        inline void add      ( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b );
        inline void add      ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );
        inline void add      ( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b );
        inline void add      ( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b );                
        // dst = a * b, elementwise
        inline void mul      ( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b );
        inline void mul      ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );
        inline void mul      ( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b );
        inline void mul      ( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b );                
        // dst = a*sa + b*sb
        inline void scale_add( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        inline void scale_add( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        inline void scale_add( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        inline void scale_add( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );                

        // dst += a*sa + b*sb
        inline void sadd__scale_add( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        inline void sadd__scale_add( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        inline void sadd__scale_add( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );
        inline void sadd__scale_add( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b, TENSOR_FLOAT sa, TENSOR_FLOAT sb );                

        // dst = a - b
        inline void sub      ( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b );
        inline void sub      ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );
        inline void sub      ( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b );
        inline void sub      ( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b );
        // dst = a + val
        inline void add      ( CTensor1D &dst, const CTensor1D &a, TENSOR_FLOAT val );
        inline void add      ( CTensor2D &dst, const CTensor2D &a, TENSOR_FLOAT val );
        inline void add      ( CTensor3D &dst, const CTensor3D &a, TENSOR_FLOAT val );
        inline void add      ( CTensor4D &dst, const CTensor4D &a, TENSOR_FLOAT val );

        // dst = a * val
        inline void mul      ( CTensor1D &dst, const CTensor1D &a, TENSOR_FLOAT val );
        inline void mul      ( CTensor2D &dst, const CTensor2D &a, TENSOR_FLOAT val );
        inline void mul      ( CTensor3D &dst, const CTensor3D &a, TENSOR_FLOAT val );
        inline void mul      ( CTensor4D &dst, const CTensor4D &a, TENSOR_FLOAT val );

        // dst += a * val
        inline void sadd__mul      ( CTensor1D &dst, const CTensor1D &a, TENSOR_FLOAT val );
        inline void sadd__mul      ( CTensor2D &dst, const CTensor2D &a, TENSOR_FLOAT val );
        inline void sadd__mul      ( CTensor3D &dst, const CTensor3D &a, TENSOR_FLOAT val );
        inline void sadd__mul      ( CTensor4D &dst, const CTensor4D &a, TENSOR_FLOAT val );

        // matrix multiplication
        // dst   = dot( a, b ) 
        // note: the 1D tensor is treated as 1 * n matrix 
        inline void dot        ( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b );    
        inline void dot        ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );            
        // dst  += dot( a, b  ) 
        inline void sadd__dot  ( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b );    
        inline void sadd__dot  ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );                    
        inline void ssub__dot  ( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b );    
        inline void ssub__dot  ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );                    
        // dst  = dot( a   ,  b.T )
        inline void dot_rt     ( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b );    
        inline void dot_rt     ( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );    
        // dst += dot( a, b.T )
        inline void sadd__dot_rt( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b );    
        inline void sadd__dot_rt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );    
        inline void ssub__dot_rt( CTensor1D &dst, const CTensor1D &a, const CTensor2D &b );    
        inline void ssub__dot_rt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );    
        // dst  = dot( a.T , b )
        inline void dot_lt      ( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b );    
        inline void sadd__dot_lt( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b );    
        inline void ssub__dot_lt( CTensor2D &dst, const CTensor1D &a, const CTensor1D &b );    
        inline void sadd__dot_lt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );    
        inline void ssub__dot_lt( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );    
    };
    
    namespace tensor{
        // ans = sum( a*b );
        TENSOR_FLOAT sum_mul( const CTensor1D &a, const CTensor1D &b );
    };

    // support for error esimtation
    namespace tensor{
        inline void sadd__abs_err( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b );
        inline void sadd__abs_err( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );
        inline void sadd__abs_err( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b );
        inline void sadd__abs_err( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b );

        inline void sadd__abs_err_rel( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b );
        inline void sadd__abs_err_rel( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );
        inline void sadd__abs_err_rel( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b );
        inline void sadd__abs_err_rel( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b );

        // calculate relative error ,ignore the case when x <= 1e-5
        inline void sadd__abs_err_relT( CTensor1D &dst, const CTensor1D &a, const CTensor1D &b );
        inline void sadd__abs_err_relT( CTensor2D &dst, const CTensor2D &a, const CTensor2D &b );
        inline void sadd__abs_err_relT( CTensor3D &dst, const CTensor3D &a, const CTensor3D &b );
        inline void sadd__abs_err_relT( CTensor4D &dst, const CTensor4D &a, const CTensor4D &b );
    };
    // support for convolutional RBM
    namespace tensor{
        namespace rbm{
            inline void sample_recified_linear( CTensor3D &mean, const CTensor3D &energy );
            inline void mean_recified_linear( CTensor3D &mean, const CTensor3D &energy );
        };
        namespace crbm{
            // normalize by maxpooling 2D
            inline void norm_maxpooling_2D( CTensor3D &mean, const CTensor3D &energy, int pool_size );

            // sample the data using 2D maxpooling 
            inline void sample_maxpooling_2D( CTensor3D &state, const CTensor3D &mean, int pool_size );

            // fit the last two dimension of src into dst's size, copy the fitted part into dst
            inline void copy_fit( CTensor2D &dst, const CTensor2D &src );
            inline void copy_fit( CTensor3D &dst, const CTensor3D &src );
            
            // fill the edge of dest by values in src
            inline void refill_edge_area( CTensor3D &dst, const CTensor3D &src, int edge_y_len, int edge_x_len );
            
            // pool up
            inline void pool_up( CTensor3D &dst , const CTensor3D &src, int pool_size ); 
                            
            // 2D convolution with bias
            // convolution, leaves the valid area
            // dst = (~a) (*)  filter + bias 
            inline void conv2_r_valid     ( CTensor3D &dst, const CTensor3D &a, const CTensor4D &filter, const CTensor1D &bias );
            
            // dst = ( a) (*) filter + bias
            inline void conv2_full        ( CTensor3D &dst, const CTensor3D &a, const CTensor4D &filter, const CTensor1D &bias );
            
            // convolution with big filter
            inline void sadd__conv2_r_big_filter( CTensor4D &dst, const CTensor3D &a, const CTensor3D &b );
            inline void ssub__conv2_r_big_filter( CTensor4D &dst, const CTensor3D &a, const CTensor3D &b );
            
            // sum over last two dimension
            inline void sadd__sum_2D( CTensor1D &dst, const CTensor3D &src );
            inline void ssub__sum_2D( CTensor1D &dst, const CTensor3D &src );

            // sum over last two dimension
            inline void sum_2D    ( CTensor2D &dst, const CTensor4D &src );            
            inline void sadd__scale( CTensor4D &dst, const CTensor2D &src, TENSOR_FLOAT scale_src );
            
            // calculate information of sparse regularization
            inline void add_sparse_info( CTensor1D &sum_mf, CTensor1D &sum_mf_grad, const CTensor3D &src, int pool_size );
        };        
    };

    // host only code
    namespace cpu_only{
        // average value
        inline TENSOR_FLOAT sum( const CTensor1D &a );
        inline TENSOR_FLOAT sum( const CTensor2D &a );
        inline TENSOR_FLOAT sum( const CTensor3D &a );
        inline TENSOR_FLOAT sum( const CTensor4D &a );
        
        // average value
        inline TENSOR_FLOAT avg( const CTensor1D &a );
        inline TENSOR_FLOAT avg( const CTensor2D &a );
        inline TENSOR_FLOAT avg( const CTensor3D &a );
        inline TENSOR_FLOAT avg( const CTensor4D &a );

        // variance
        inline TENSOR_FLOAT var( const CTensor1D &a );
        inline TENSOR_FLOAT var( const CTensor2D &a );
        inline TENSOR_FLOAT var( const CTensor3D &a );
        inline TENSOR_FLOAT var( const CTensor4D &a );
        
        // standard variance
        inline TENSOR_FLOAT std_var( const CTensor1D &a );
        inline TENSOR_FLOAT std_var( const CTensor2D &a );
        inline TENSOR_FLOAT std_var( const CTensor3D &a );
        inline TENSOR_FLOAT std_var( const CTensor4D &a );
        
        // min value
        inline TENSOR_FLOAT min_value( const CTensor1D &a );
        inline TENSOR_FLOAT min_value( const CTensor2D &a );
        inline TENSOR_FLOAT min_value( const CTensor3D &a );
        inline TENSOR_FLOAT min_value( const CTensor4D &a );
        // max value
        inline TENSOR_FLOAT max_value( const CTensor1D &a );
        inline TENSOR_FLOAT max_value( const CTensor2D &a );
        inline TENSOR_FLOAT max_value( const CTensor3D &a );
        inline TENSOR_FLOAT max_value( const CTensor4D &a );

        // random extract a region from src
        inline void rand_extract( CTensor3D &dst, const CTensor3D &src );
        // shuffle the h_index of data randomly
        inline void shuffle( CTensor4D &data );
        inline void shuffle( std::vector<CTensor3D> &data );
    };
};

#include "apex_tensor_cpu_inline.h"

#endif


