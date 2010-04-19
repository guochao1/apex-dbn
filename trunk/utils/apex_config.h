#ifndef _APEX_CONFIG_H_
#define _APEX_CONFIG_H_
#define _CRT_SECURE_NO_WARNINGS

#include "apex_utils.h"
#include <cstdio>

/* simple helper for load in configure files */
namespace apex_utils{    
    /* load in config file */
    class ConfigIterator{
    private:
        FILE *fi;        
        char ch_buf;
        char s_name[256],s_val[256],s_buf[246];

        inline void skip_line(){           
            do{
               ch_buf = fgetc( fi );
            }while( ch_buf != EOF && ch_buf != '\n' && ch_buf != '\r' );
        }
        
        inline void parse_str( char tok[] ){
            int i = 0; 
            while( (ch_buf = fgetc(fi)) != EOF ){
                switch( ch_buf ){
                case '\\': tok[i++] = fgetc( fi ); break;
                case '\"': tok[i++] = '\0'; 
						return;
                case '\r':
                case '\n': apex_utils::error("unterminated string"); break;
                default: tok[i++] = ch_buf;
                }
            }
            apex_utils::error("unterminated string"); 
        }
        // return newline 
        inline bool get_next_token( char tok[] ){
            int i = 0;
            bool new_line = false; 
            while( ch_buf != EOF ){
                switch( ch_buf ){
                case '#' : skip_line(); new_line = true; break;
                case '\"':
                    if( i == 0 ){
                        parse_str( tok );ch_buf = fgetc(fi); return new_line;
                    }else{
                        apex_utils::error("token followed directly by string"); 
                    }
                case '=':
					if( i == 0 ) {
						ch_buf = fgetc( fi );     
                        tok[0] = '='; 
                        tok[1] = '\0'; 
                    }else{
                        tok[i] = '\0'; 
                    }
					return new_line;
                case '\r':
                case '\n':
					if( i == 0 ) new_line = true;
                case ' ' :
                    ch_buf = fgetc( fi );
                    if( i > 0 ){
                        tok[i] = '\0'; 
                        return new_line;
                    }               
					break;
                default: 
                    tok[i++] = ch_buf;
                    ch_buf = fgetc( fi );
                    break;                    
				}
			}
			return true;
		}

    public:
        ConfigIterator( const char *fname ){
            fi = apex_utils::fopen_check( fname, "r");
            ch_buf = fgetc( fi );
        }
        ~ConfigIterator(){
            fclose( fi );
        }
        inline const char *name()const{
            return s_name;
        }
        inline const char *val() const{
            return s_val;
        }
        inline bool next(){            
            while( !feof( fi ) ){
                get_next_token( s_name );

                if( s_name[0] == '=')  return false;               
				if( get_next_token( s_buf ) || s_buf[0] != '=' ) return false;			   				
				if( get_next_token( s_val ) || s_val[0] == '=' ) return false;
                return true;
            }
            return false;
        }        
    };

};

#endif


