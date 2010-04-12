#ifndef _CUDA_SAMPLING_CUH_
#define _CUDA_SAMPLING_CUH_
/* methods for sampling */
namespace cuda_rand{
    /* 
       sample binary distribution given prob and rnd,
       rnd is generated uniform in [0,1)
    */
    inline __device__ float sample_binary( float prob, float rnd01 ){
        return prob > rnd01 ? 1.0f : 0.0f; 
    }
    
    __device__ void __box_muller(float &u1, float &u2){
        float   r = sqrtf(-2.0f * logf(u1));
        float phi = 2.0f * 3.14159265358979f * u2;
        u1 = r * __cosf(phi);
        u2 = r * __sinf(phi);
    }
    
    // transofrm r in [1,2) into (0,1]
    inline __device__ float __to_upper_uniform( float r ){
        return  2.0f - r;
    }

    // sample standard normal distribution
    inline __device__ void sample_standard_gaussian( float &tx, float &ty ){
        // do box muller
        tx =  __to_upper_uniform( tx );
        ty =  __to_upper_uniform( ty );

        __box_muller( tx, ty );        
    } 

    /* sample gaussian random variable, pairs the threads and do box muller */
    inline __device__ void sample_standard_gaussian_by_pair( float s_rnd[], int tid ){
         if( (tid & 1) == 0 ){
             float rx = s_rnd[tid];
             float ry = s_rnd[tid+1];
             sample_standard_gaussian( rx, ry );             
             s_rnd[tid]   = rx; 
             s_rnd[tid+1] = ry;                         
        }
    }

    template<int block_dim_bits>
    inline __device__ float sample_gaussian( float rnd, int tid, float s_rnd[1<<block_dim_bits] ){
        // get gaussian random variable
        s_rnd[ tid ] = rnd; 
        __syncthreads();
        cuda_rand::sample_standard_gaussian_by_pair( s_rnd, tid );
        __syncthreads();
        return s_rnd[ tid ];
    }
    
    
    template<int pool_bits>
    inline __device__ void sample_maxpooling( int y_start, int x_start, float s_mm[16][16], float rnd01 ){
        float sum = 0.0f;
        bool  hit = false;
        for( int y = y_start ; y < y_start + (1<<pool_bits) ; y ++ )
            for( int x = x_start ; x < x_start + (1<<pool_bits) ; x ++ ){
                sum += s_mm[y][x];
                if( sum > rnd01 && !hit ){
                    hit = true; s_mm[y][x] = 1.0f;
                } else{
                    s_mm[y][x] = 0.0f;
                }
            }
    }

    template<int pool_size>
    inline __device__ void sample_maxpooling_ord( int y_start, int x_start, float s_mm[pool_size][pool_size*16], float rnd01 ){
        float sum = 0.0f;
        bool  hit = false;
        for( int y = y_start ; y < y_start + pool_size ; y ++ )
            for( int x = x_start ; x < x_start + pool_size ; x ++ ){
                sum += s_mm[y][x];
                if( sum > rnd01 && !hit ){
                    hit = true; s_mm[y][x] = 1.0f;
                } else{
                    s_mm[y][x] = 0.0f;
                }
            }
    }

};

#endif

