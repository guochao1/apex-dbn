#ifndef _APEX_TASK_H_
#define _APEX_TASK_H_
namespace apex_utils{
    /* interface of task program */
    class ITask{
    public:
        virtual void set_param( const char *name , const char *val ) = 0;
        virtual void set_task ( const char *task ) =0;
        virtual void print_task_help( FILE *fo ) const = 0;
        virtual void run_task( void )= 0;
    public:
        virtual ~ITask(){}
    };
    
    inline int run_task( int argc , char *argv[] , ITask *tsk ){
        if( argc < 2 ){
            tsk->print_task_help( stdout );
            return 0;            
        }   
        tsk->set_task( argv[1] );
        
        for( int i = 2 ; i < argc ; i ++ ){
            char name[256],val[256];
            if( sscanf( argv[i] ,"%[^=]=%[^\n]", name , val ) == 2 ){
                tsk->set_param( name , val );
		}   
        }
        tsk->run_task();
        return 0;
    } 
};
#endif
