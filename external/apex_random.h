#ifndef _APEX_RANDOM_H_
#define _APEX_RANDOM_H_

/*   generate random number using Mersene Twister */

extern "C"{
#include "dSFMT/dSFMT.h"
}

#include <cmath>

/*
  self defined random number generator,
  we base our generator on good generators
  
  author: tqchen
*/

namespace apex_random{

	/*-------interface code------*/
	inline void seed( uint32_t seed ){
		dsfmt_gv_init_gen_rand( seed );
	}
		
	/* return a 32 bit random number */
	inline uint32_t next_uint32(){
		return dsfmt_gv_init_gen_rand();
	}
	
	/*-------------------------------*/

	/* return a random number in n */
	inline uint32_t next_uint32( uint32_t n ){
		return (uint32_t) ( next_double() * n ) ;
	}
	
	/* return a real number uniform in [0,1) */
	inline double next_double(){
		return dsfmt_gv_genrand_close_open();
	}
	/* return a real numer uniform in (0,1) */
    inline double next_double2(){
        return dsfmt_gv_genrand_open_open();
    }

	/* return  x~N(0,1) */
	inline double sample_normal(){
		double x,y,s;
		do{
			x = 2 * next_double2() - 1.0;
			y = 2 * next_double2() - 1.0;
			s = x*x + y*y;
		}while( s >= 1.0 || s == 0.0 );
		
		return x * sqrt( -2.0 * log(s) / s ) ;
	}
	
	/* return iid x,y ~N(0,1) */
	inline void sample_normal2D( double &xx, double &yy ){
		double x,y,s;
		do{
			x = 2 * next_double2() - 1.0;
			y = 2 * next_double2() - 1.0;
			s = x*x + y*y;
		}while( s >= 1.0 || s == 0.0 );
		double t = sqrt( -2.0 * log(s) / s ) ;
		xx = x * t; 
		yy = y * t;
	}
	
	inline double sample_normal( double mu, double sigma ){
		return sample_normal() * sigma + mu;
	}

	/* return 1 with probability p, coin flip */
	inline int sample_binary( double p ){
		return next_double() <  p;  
	}
};

#endif
