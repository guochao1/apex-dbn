#ifndef _APEX_UTILS_H_
#define _APEX_UTILS_H_

#include <cstdio>
#include <cstdlib>

namespace apex_utils{        
    inline void error( const char *msg ){
        printf("assert error:%s\n",msg );
        exit( -1 );
    }

    inline void warning( const char *msg ){
        printf("warning:%s\n",msg );
    }
    
    inline FILE *fopen_check( const char *fname , const char *flag ){
		FILE *fp = fopen( fname , flag );
		if( fp == NULL ){
			printf("can not open file \"%s\"\n",fname );
			exit( -1 );
		}
		return fp;
 	} 
    
    namespace iterator{    
        // interface of abstract iterator
        template<typename T>
        class IIterator{
        public:
            // set the parameter
            virtual void set_param( const char *name, const char *val ) = 0;
            // initalize the iterator so that we can use the iterator
            virtual void init( void ) = 0;
            // destroy the iterator, we can no longer use the iterator anymore
            virtual void destroy( void ) = 0;
            // set before first of the item
            virtual void before_first() = 0;
            // move to next item
            virtual bool next() = 0;
            // get current matrix 
            virtual const T value() const = 0;
        public:
            virtual ~IIterator(){}
        };
    };    
};

#endif
