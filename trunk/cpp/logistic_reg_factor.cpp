// use matrix factorization to do 
// factored version of logistic regression

#include "../utils/apex_utils.h"
#include "../utils/apex_config.h"
#include "../tensor/apex_tensor.h"
#include <vector>
#include <cstring>
#include <cstdlib>

using namespace std;
using namespace apex_tensor;

struct LogRegParam{
    float learning_rate, learning_rate_B, learning_rate_bias;
    float wd, wd_B, momentum;
    
    float q_init_sigma,p_init_sigma,b_init_sigma;

    int   num_iter;
    int   batch_size;
    int   print_step;
    int   dump_step;

    int   num_factor;
    
    int   num_user_f;
    int   num_item_f;
    int   num_global_f;

    char  name_model_in[256];
    char  name_model_folder[256];    
    char  name_user_f[256];
    char  name_item_f[256];
    char  name_global_f[256];
    char  name_rank[256];
    
    LogRegParam(){
        learning_rate_B = learning_rate_bias = learning_rate = 0.01;
        wd_B = wd = 1.0; momentum = 0.5;
        q_init_sigma = 0.001; 
        p_init_sigma = 0.001;
        b_init_sigma = 0;
        num_iter = 10; 
        batch_size = 1000;
        print_step = 1;
        dump_step  = 100; 
        strcpy( name_model_folder, "models" );
        strcpy( name_model_in    , "NULL"   );
        strcpy( name_user_f    , "user_f.txt"   );
        strcpy( name_item_f    , "item_f.txt"   );
        strcpy( name_global_f  , "global_f.txt" );
        strcpy( name_rank      , "rank.txt"     );
    }
    inline void set_param( const char *name, const char *val ){
        if( !strcmp( name,"learning_rate") ) learning_rate = atof( val );
        if( !strcmp( name,"learning_rate_B") ) learning_rate_B = atof( val );
        if( !strcmp( name,"learning_rate_bias") ) learning_rate_bias = atof( val );
        if( !strcmp( name,"wd") )            wd = atof( val );
        if( !strcmp( name,"wd_B") )          wd_B = atof( val );
        if( !strcmp( name,"momentum") )      momentum = atof( val );
        if( !strcmp( name,"q_init_sigma") )  q_init_sigma = atof( val );
        if( !strcmp( name,"p_init_sigma") )  p_init_sigma = atof( val );
        if( !strcmp( name,"b_init_sigma") )  b_init_sigma = atof( val );        
        if( !strcmp( name,"num_iter") )      num_iter = atoi( val );
        if( !strcmp( name,"batch_size") )    batch_size = atoi( val );
        if( !strcmp( name,"print_step") )    print_step = atoi( val );
        if( !strcmp( name,"dump_step") )     dump_step = atoi( val );
        if( !strcmp( name,"num_factor") )    num_factor = atoi( val );
        if( !strcmp( name,"num_user_f") )    num_user_f = atoi( val );
        if( !strcmp( name,"num_item_f") )    num_item_f = atoi( val );
        if( !strcmp( name,"num_global_f") )  num_global_f = atoi( val );
        if( !strcmp( name,"model_in") )      strcpy( name_model_in, val );
        if( !strcmp( name,"model_folder") )  strcpy( name_model_folder, val );
        if( !strcmp( name,"file_user_f") )   strcpy( name_user_f, val );
        if( !strcmp( name,"file_item_f") )   strcpy( name_item_f, val );
        if( !strcmp( name,"file_global_f") ) strcpy( name_global_f, val );
        if( !strcmp( name,"file_rank") )     strcpy( name_rank, val );
    }
};

inline TENSOR_FLOAT sigmoid( TENSOR_FLOAT d ){
    return 1 / (1 + exp(-d) ); 
}

inline void save_model( const char *fname, const CTensor2D &Q, const CTensor2D &P, const CTensor1D &B, TENSOR_FLOAT bias ){
    FILE *fo = apex_utils::fopen_check( fname,"wb" );
    tensor::save_to_file( Q , fo );
    tensor::save_to_file( P , fo );
    tensor::save_to_file( B , fo );
    fwrite( &bias, sizeof(TENSOR_FLOAT), 1, fo );
    fclose( fo );
}
inline void load_model( const char *fname, CTensor2D &Q, CTensor2D &P, CTensor1D &B, TENSOR_FLOAT &bias ){
    FILE *fi = apex_utils::fopen_check( fname,"rb" );
    tensor::load_from_file( Q , fi );
    tensor::load_from_file( P , fi );
    tensor::load_from_file( B , fi );
    if( fread( &bias , sizeof(TENSOR_FLOAT), 1,fi) <= 0 )
        apex_utils::error("load model logistic_regression");
    fclose( fi );
}

inline void save_model( int idx, const LogRegParam &param,
                        const CTensor2D &Q, const CTensor2D &P, const CTensor1D &B, TENSOR_FLOAT bias ){
    char name[256];
    sprintf( name, "%s/%d.model", param.name_model_folder, idx );
    save_model( name, Q, P, B, bias );
}

// B: 1 * v   Q: n * k, P m * k 
inline void logistic_regression( CTensor2D &Q, CTensor2D &P, CTensor1D &B, TENSOR_FLOAT &bias,                               
                                 const vector<CTensor1DSparse> &user_f,
                                 const vector<CTensor1DSparse> &item_f,
                                 const vector<CTensor1DSparse> &global_f,                          
                                 const vector<int> &rank,
                                 const LogRegParam &param ){
    int sample_counter = 0;
    CTensor2D dQ; 
    CTensor2D dP; 
    CTensor1D dB; 
    TENSOR_FLOAT dbias = 0;
    int num_factor = Q.x_max;    
    CTensor1D prjQ(num_factor), prjP(num_factor);

    dQ = alloc_like( Q );
    dP = alloc_like( P );
    dB = alloc_like(B);
    tensor::alloc_space( prjQ );
    tensor::alloc_space( prjP );
    
    dQ = 0; dP = 0; dB = 0;
    
    TENSOR_FLOAT sum_likelihood = 0;    
    
    for( int iter = 0 ; iter < param.num_iter ; iter++ ){
        for( size_t i = 0 ; i < rank.size() ; i ++ ){
            // prjQ,P : 1 * k
            prjQ = dot( user_f[i], Q );
            prjP = dot( item_f[i], P );

            // calculate energy
            TENSOR_FLOAT energy = tensor::sum_mul( prjP, prjQ ) + tensor::sum_mul( global_f[i], B ) + bias;
            TENSOR_FLOAT pred   = sigmoid( energy * rank[i] );

            // calculate update
            prjQ *= ( 1 - pred );  
            prjP *= ( 1 - pred );
            
            dQ +=  dot( user_f[i].T(), prjP );
            dP +=  dot( item_f[i].T(), prjQ );
            dB +=  global_f[i] * ( 1 - pred );
            dbias += 1 - pred;
            
            sum_likelihood      += pred;

            if( ++ sample_counter % param.batch_size == 0 ){
                // update parameter 
                Q += ( dQ -= param.wd*Q )   * param.learning_rate;
                P += ( dP -= param.wd*P )   * param.learning_rate;
                B += ( dB -= param.wd_B*B ) * (param.learning_rate_B / sample_counter);

                bias += dbias * param.learning_rate_bias / sample_counter;

                dQ *= param.momentum; 
                dP *= param.momentum; 
                dB *= param.momentum; 
                dbias *= param.momentum;
                sample_counter = 0; 
            }                        
        }
        if( (iter+1) % param.print_step == 0 )
            printf("%d iter, likelihood=%lf\n", iter, (double)sum_likelihood/rank.size() );
        if( (iter+1) % param.dump_step == 0 )
            save_model( (iter+1)/param.dump_step , param, Q, P, B, bias );
        sum_likelihood = 0;                
    }
    
    tensor::free_space( prjQ );
    tensor::free_space( prjP );
    tensor::free_space( dQ );
    tensor::free_space( dP );
    tensor::free_space( dB );    
}

inline void load_data( vector<CTensor1DSparse> &feature, const char *fname ){
    FILE *fi = apex_utils::fopen_check( fname, "r" );
    int x,y,lastx = -1;
    float v;
    vector<int>   idx;
    vector<float> vals;
    
    while( fscanf( fi, "%d%d%f",&x,&y,&v ) == 3 ){
        if( x != lastx ){
            lastx = x; 
            if( idx.size() > 0 ){
                feature.push_back( tensor::create_sparse( idx, vals ) );
            }
            idx.clear(); vals.clear();
        }
        idx.push_back( y-1 );
        vals.push_back( v );
    }
    if( idx.size() > 0 ){
        feature.push_back( tensor::create_sparse( idx, vals ) );
    }
    idx.clear(); vals.clear();
    
    fclose( fi );
} 
 
inline void load_data( vector<int> &rank, const char *fname ){
    int r;
    FILE *fi = apex_utils::fopen_check( fname, "r" );    
    while( fscanf( fi, "%d",&r ) == 1 ){
        rank.push_back( r == 0 ? -1: r );
    }
    fclose( fi );
}

inline void free_space( vector<CTensor1DSparse> &feature ){
    for( size_t i = 0; i < feature.size() ; i ++ )
        tensor::free_space( feature[i] );
    feature.clear();
}

inline void train_proc( const LogRegParam &param ){
    CTensor2D Q( param.num_user_f, param.num_factor );
    CTensor2D P( param.num_item_f, param.num_factor );
    CTensor1D B( param.num_global_f );
    TENSOR_FLOAT bias;
    tensor::alloc_space( Q );
    tensor::alloc_space( P );
    tensor::alloc_space( B );

    if( !strcmp( param.name_model_in,"NULL" )  ){
        tensor::sample_gaussian( Q, param.q_init_sigma );
        tensor::sample_gaussian( P, param.p_init_sigma );
        tensor::sample_gaussian( B, param.b_init_sigma );
        bias = 0;
    }else{
        load_model( param.name_model_in, Q, P, B, bias );
    }
    vector<int> rank;
    vector<CTensor1DSparse> user_f,item_f,global_f;

    load_data( rank    , param.name_rank );
    load_data( user_f  , param.name_user_f );
    load_data( item_f  , param.name_item_f );
    load_data( global_f, param.name_global_f );
    if( user_f.size() != rank.size()  ||
        user_f.size() != item_f.size()||
        user_f.size() != global_f.size() ){
        apex_utils::error("feature number not matched\n");
    }

    printf("data loaded\n");
    logistic_regression( Q,P,B, bias, user_f,item_f,global_f,rank,param );
    
    printf("training end..\n");
    save_model( 0, param, Q,P,B, bias );
    
    free_space( user_f );
    free_space( item_f );
    free_space( global_f );
    tensor::free_space ( Q );
    tensor::free_space ( P );
    tensor::free_space ( B );
}

int main( int argc, char *argv[] ){
    if( argc < 2 ){
        printf("Usage:<config>\n"); return 0;
    }

    LogRegParam param;
    apex_utils::ConfigIterator cfg( argv[1] );
    apex_tensor::init_tensor_engine_cpu( 0 );

    while( cfg.next() ){
        param.set_param( cfg.name(), cfg.val() );

    }
    printf("configure end, start training...\n");
    train_proc( param );
    apex_tensor::destroy_tensor_engine_cpu();
    return 0;
}

