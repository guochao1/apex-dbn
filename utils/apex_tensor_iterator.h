#ifndef _APEX_TENSOR_ITERATOR_H_
#define _APEX_TENSOR_ITERATOR_H_

#include "../tensor/apex_tensor.h"

namespace apex_utils{        
    // tensor iterator that iterates over the data      
    class ITensor1DIterator{
    public:
        // set the parameter
        virtual void set_param( const char *name, const char *val )=0;
        // initalize the iterator
        virtual void init( void ) = 0;
        // move to next mat trunk
        virtual bool next_trunk() = 0;
        // get current matrix 
        virtual const apex_tensor::CTensor2D trunk() const = 0;
        // get validation trunk
		virtual const apex_tensor::CTensor2D validation_trunk() const = 0;
        // set before first of the item
        virtual void before_first() = 0;
    };
    
};

#endif
