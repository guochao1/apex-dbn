#ifndef _APEX_UTILS_H_
#define _APEX_UTILS_H_

#include <cstdio>
#include <cstdlib>

namespace apex_utils{        
    inline void error( const char *msg ){
        printf("assert error:%s\n",msg );
        exit( -1 );
    }
    
    inline void assert_true( bool exp, const char *msg ){
        if( !exp ) error( msg );
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
            // set before first of the item
            virtual void before_first() = 0;
            // move to next item
            virtual bool next() = 0;
            // get current matrix 
            virtual const T &value() const = 0;
        public:
            virtual ~IIterator(){}
        };
    };    

    // iterator with limit counter
    namespace iterator{
        // iterator with limit counter
        template<typename T>
        class LimitCounterIterator:public IIterator<T>{
        private: 
            IIterator<T> *base_itr;
            int limit_max, counter;
        public:            
            LimitCounterIterator( IIterator<T> *base_itr, int limit_max ){
                this->base_itr  = base_itr;
                this->limit_max = limit_max;
                this->before_first();
            } 
            virtual void set_param( const char *name, const char *val ){}
            virtual void init( void ){}
            virtual void before_first(){
                base_itr->before_first();
                counter = limit_max;
            }
            virtual bool next(){
                if( counter > 0 ){
                    counter -- ;
                    return base_itr->next();
                }else{
                    return false;
                }
            }
            virtual const T &value() const{
                return base_itr->value();
            }
        };       
    };
};

#endif
