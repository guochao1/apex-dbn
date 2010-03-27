#ifndef _CUDA_TENSOR_OP_CU_
#define _CUDA_TENSOR_OP_CU_

#include "cuda_tensor.cuh"
namespace apex_tensor{
    namespace cuda_tensor{
        template<int st_m,int block_dim_bits>
        __global__ void store_kernel( float *elem , 
                                      size_t pitch,
                                      int    y_max, int x_max,
                                      float  src ){
            const int tid     = (blockIdx.x << block_dim_bits) + threadIdx.x;            
            const int x_mm    = get_align_width( x_max );
            const int y       = tid / x_mm;
            const int x       = tid % x_mm;

            elem = get_line( elem, y, pitch );  
                        
            if( y < y_max  && x < x_max ){
                store_method::__store<st_m>( elem[ x ] , src );
            }            
        }

        // store one element to another 
        template<int st_m,typename T>
        inline void store( T &ts, float src ){
            int stride = get_align_width( ts.x_max );
            int y_max  = num_line( ts );
            int x_max  = ts.x_max;
            
            int num_block = (y_max*stride + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , 1, 1 );
            
            store_kernel<st_m,BASE_THREAD_BITS> <<<dimGrid,dimBlock,0,cuda_async::get_stream(ts)>>>
                ( ts.elem, ts.pitch, y_max, x_max, src );
        }  
        
        // test pass
        template<int st_m,int mapm_A,int block_dim_bits>
        __global__ void map_A_kernel( float *elem_dst , const float *elem_src, 
                                      size_t pitch_dst, size_t pitch_src,
                                      int y_max       , int x_max ){
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;
            const int x_mm= get_align_width( x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            
            elem_dst = get_line      ( elem_dst, y, pitch_dst );
            elem_src = get_line_const( elem_src, y, pitch_src );
            
            
            if( y < y_max && x < x_max ){
                float val = map_method_A::__map<mapm_A>( elem_src[x] );
                store_method::__store<st_m>( elem_dst[x], val );
            }            
        }
        
        template<int st_m,int mapm_A,typename T>
        inline void map_A( T &dst, const T &src ){
            int stride     = get_align_width( dst.x_max );
            int y_max      = num_line( dst );
            int x_max      = dst.x_max;
            
            int num_block = (y_max*stride + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , 1, 1 );
            
            map_A_kernel<st_m,mapm_A,BASE_THREAD_BITS> <<<dimGrid,dimBlock,0,cuda_async::get_stream(dst,src)>>>
                ( dst.elem, src.elem, dst.pitch,  src.pitch, y_max, x_max );
        } 
                
        // takes a source and a float 
        template<int st_m,int mapm_B,int block_dim_bits>
        __global__ void map_B_kernel( float *elem_dst , const float *elem_src, 
                                      size_t pitch_dst, size_t pitch_src,
                                      int y_max       , int x_max, float src_b ){
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;
            const int x_mm= get_align_width( x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            
            elem_dst = get_line      ( elem_dst, y, pitch_dst );
            elem_src = get_line_const( elem_src, y, pitch_src );
            
            
            if( y < y_max  && x < x_max ){
                float val = map_method_B::__map<mapm_B>( elem_src[x], src_b );
                store_method::__store<st_m>( elem_dst[x] , val );
            }            
        }
        
        template<int st_m,int mapm_B,typename T>
        inline void map_B( T &dst, const T &src, float src_b ){
            int stride     = get_align_width( dst.x_max );
            int y_max      = num_line( dst );
            int x_max      = dst.x_max;
            
            int num_block = (y_max*stride + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , 1, 1 );
            
            map_B_kernel<st_m,mapm_B,BASE_THREAD_BITS> <<<dimGrid,dimBlock,0,cuda_async::get_stream(dst,src)>>>
                ( dst.elem, src.elem, dst.pitch,  src.pitch, y_max, x_max, src_b );
        }          

        // takes two source and a float 
        template<int st_m,int mapm_B,int block_dim_bits>
        __global__ void map_C_kernel( float *elem_dst , 
                                      const float *elem_srca, const float *elem_srcb, 
                                      size_t pitch_dst, size_t pitch_srca, size_t pitch_srcb,
                                      int y_max       , int x_max ){
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;
            const int x_mm= get_align_width( x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            
            elem_dst  = get_line       ( elem_dst, y, pitch_dst );
            elem_srca = get_line_const( elem_srca, y, pitch_srca );
            elem_srcb = get_line_const( elem_srcb, y, pitch_srcb );
            
            
            if( y < y_max  && x < x_max ){
                float val = map_method_B::__map<mapm_B>( elem_srca[x], elem_srcb[x] );
                store_method::__store<st_m>( elem_dst[x] , val );
            }            
        }
        
        template<int st_m,int mapm_B,typename T>
        inline void map_C( T &dst, const T &srca, const T &srcb ){
            int stride     = get_align_width( dst.x_max );
            int y_max      = num_line( dst );
            int x_max      = dst.x_max;
            
            int num_block = (y_max*stride + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , 1, 1 );
            
            map_C_kernel<st_m,mapm_B,BASE_THREAD_BITS> <<<dimGrid,dimBlock,0,cuda_async::get_stream(dst,srca,srcb)>>>
                ( dst.elem, srca.elem, srcb.elem, dst.pitch, srca.pitch, srcb.pitch, y_max, x_max );
        }          

        // takes two source and a float 
        template<int st_m,int mapm_D,int block_dim_bits>
        __global__ void map_D_kernel( float *elem_dst , 
                                      const float *elem_srca, const float *elem_srcb, 
                                      size_t pitch_dst, size_t pitch_srca, size_t pitch_srcb,
                                      int y_max       , int x_max,
                                      float sa        , float sb   ){
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;
            const int x_mm= get_align_width( x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;
            
            elem_dst  = get_line       ( elem_dst, y, pitch_dst );
            elem_srca = get_line_const( elem_srca, y, pitch_srca );
            elem_srcb = get_line_const( elem_srcb, y, pitch_srcb );
            
            
            if( y < y_max  && x < x_max ){
                float val = map_method_D::__map<mapm_D>( elem_srca[x], elem_srcb[x], sa, sb );
                store_method::__store<st_m>( elem_dst[x] , val );
            }            
        }
        
        template<int st_m,int mapm_D,typename T>
        inline void map_D( T &dst, const T &srca, const T &srcb, float sa, float sb ){
            int stride     = get_align_width( dst.x_max );
            int y_max      = num_line( dst );
            int x_max      = dst.x_max;
            
            int num_block = (y_max*stride + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , 1, 1 );
            
            map_D_kernel<st_m,mapm_D,BASE_THREAD_BITS> <<<dimGrid,dimBlock,0,cuda_async::get_stream(dst,srca,srcb)>>>
                ( dst.elem, srca.elem, srcb.elem, dst.pitch, srca.pitch, srcb.pitch, y_max, x_max, sa, sb );
        }                  

        
        template<int st_m,int mapm_B,int block_dim_bits >
        __global__ void map_E_kernel( __GT4D dst,
                                      const __GT2D src, float src_b ){
            const int block_y = blockIdx.y / dst.z_max;
            const int block_x = blockIdx.y % dst.z_max;
        
            __shared__ float src_a;
            if( threadIdx.x == 0 ){
                src_a = src[block_y][block_x];
            }
            __GT2D dd = dst[block_y][block_x];
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;
            const int x_mm= get_align_width( dd.x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;                                                
            
            __syncthreads();

            if( y < dd.y_max  && x < dd.x_max ){
                float val = map_method_B::__map<mapm_B>( src_a, src_b );
                store_method::__store<st_m>( dd[y][x] , val );
            }            
        }
        
        template<int st_m,int mapm_B>
        inline void map_E( GTensor4D &dst, const GTensor2D &src, float src_b ){
            int stride    = get_align_width( dst.x_max );            
            int num_block = (dst.y_max*stride + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , dst.z_max*dst.h_max, 1 );
            
            map_E_kernel<st_m,mapm_B,BASE_THREAD_BITS> <<<dimGrid,dimBlock,0,cuda_async::get_stream(dst,src)>>>
                ( __GT(dst), __GT(src), src_b );
        }          
        
        
        // map the edge area only
        template<int st_m,int mapm_A,int block_dim_bits>
        __global__ void map_A_edge_kernel( __GT3D dst , const __GT3D src, 
                                           int edge_y_len, int edge_x_len ){
            __GT2D dd = dst[ blockIdx.y ];
            const __GT2D ss = src[ blockIdx.y ];
            const int tid = (blockIdx.x << block_dim_bits) + threadIdx.x;
            const int x_mm= get_align_width( dd.x_max );
            const int y   = tid / x_mm;
            const int x   = tid % x_mm;

            if( y < dd.y_max && x < dd.x_max )
                if( y < edge_y_len || y >= dd.y_max - edge_y_len ||
                    x < edge_x_len || x >= dd.x_max - edge_x_len  ){
                        float val = map_method_A::__map<mapm_A>( ss[y][x] );
                        store_method::__store<st_m>( dd[y][x], val );
                }            
        }
        
        template<int st_m,int mapm_A>
        inline void map_A_edge( GTensor3D &dst, const GTensor3D &src, int edge_y_len, int edge_x_len ){
            int stride     = get_align_width( dst.x_max );            
            int num_block = (dst.y_max*stride + BASE_THREAD_NUM-1)/BASE_THREAD_NUM;

            dim3 dimBlock( BASE_THREAD_NUM, 1, 1 );
            dim3 dimGrid ( num_block      , dst.z_max , 1 );
            
            map_A_edge_kernel<st_m,mapm_A,BASE_THREAD_BITS> <<<dimGrid,dimBlock,0,cuda_async::get_stream(dst,src)>>>
                ( __GT(dst), __GT(src), edge_y_len, edge_x_len );
        } 
    };
};

#endif
