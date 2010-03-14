#ifndef _APEX_UTILS_H_
#define _APEX_UTILS_H_

#include <cstdio>
#include <cstdlib>

namespace apex_utils{        
    inline void error( const char *msg ){
        printf("assert error:%s\n",msg );
        exit( -1 );
    }

    inline FILE *fopen_check( const char *fname , const char *flag ){
		FILE *fp = fopen( fname , flag );
		if( fp == NULL ){
			printf("can not open file \"%s\"\n",fname );
			exit( -1 );
		}
		return fp;
 	}       
};

#endif
