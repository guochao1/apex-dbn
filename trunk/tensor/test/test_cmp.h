#ifndef _TEST_CMP_H_
#define _TEST_CMP_H_

#include "../apex_tensor.h"
#include "test_stats.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
using namespace apex_tensor;

void test_gaussian( int num_iter ){
    TTensor4D tg_f  ( V_MAX, H_MAX,  F_Y_MAX, F_X_MAX );
    CTensor4D tc_f  ( V_MAX, H_MAX,  F_Y_MAX, F_X_MAX );
    
    tensor::alloc_space( tg_f );
    tensor::alloc_space( tc_f );
    printf("start test sample gaussian\n");
    
    double mean=0.0, var=0.0;

    for( int i = 0 ; i < num_iter ; i ++ ){
		printf("\r                                  \r");
		printf("round [%8d]", i);
        fflush( stdout );

		tensor::sample_gaussian( tg_f, sd ); 
        tensor::copy( tc_f, tg_f );
        mean += (double)apex_tensor::cpu_only::avg( tc_f );
        var  += (double)apex_tensor::cpu_only::var( tc_f );
    } 
    printf("\nmean=%lf, sd_var=%lf, sd=%f\n" , mean/num_iter, sqrt( var/num_iter ), sd ); 
    
    tensor::free_space( tg_f );
    tensor::free_space( tc_f );
}

void test_conv2_r_valid( int num_iter ){
    TTensor1D tg_hb ( H_MAX );
    TTensor3D tg_v  ( V_MAX, V_Y_MAX, V_X_MAX );
    TTensor3D tg_h  ( H_MAX, H_Y_MAX, H_X_MAX );
    TTensor4D tg_f  ( V_MAX, H_MAX,  F_Y_MAX, F_X_MAX );
 
	CTensor1D tc_hb ( H_MAX );
    CTensor3D tc_v  ( V_MAX, V_Y_MAX, V_X_MAX );
    CTensor3D tc_h  ( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor3D tc_h_g( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor4D tc_f  ( V_MAX, H_MAX,  F_Y_MAX, F_X_MAX );

	tensor::alloc_space( tg_v );
    tensor::alloc_space( tg_hb );
    tensor::alloc_space( tg_h );
    tensor::alloc_space( tg_f );
    tensor::alloc_space( tc_v );
    tensor::alloc_space( tc_hb );
    tensor::alloc_space( tc_h );
    tensor::alloc_space( tc_h_g );
    tensor::alloc_space( tc_f );
    printf("start test conv2_r_valid\n");
    
	TestStats<CTensor3D> stats( "conv2_r_valid_CPU","conv2_r_valid_GPU");
    stats.abs_err.set_param( H_MAX, H_Y_MAX, H_X_MAX );
    stats.abs_err_rel.set_param( H_MAX, H_Y_MAX, H_X_MAX );
    stats.abs_err_relT.set_param( H_MAX, H_Y_MAX, H_X_MAX );
    stats.init();
    
    for( int i = 0 ; i < num_iter ; i ++ ){
		printf("\r                                  \r");
		printf("round [%8d]", i);
        fflush( stdout );

		tensor::sample_gaussian( tc_f, sd ); 
		tensor::sample_gaussian( tc_v, sd );
		tensor::sample_gaussian( tc_hb, sd );

        double c_start = clock();
        tensor::crbm::conv2_r_valid( tc_h, tc_v, tc_f, tc_hb );
        stats.time_A += (clock() - c_start) / CLOCKS_PER_SEC;

        tensor::copy( tg_f, tc_f );
        tensor::copy( tg_v, tc_v );        
        tensor::copy( tg_hb, tc_hb );
        
        double g_start = clock();
        tensor::crbm::conv2_r_valid( tg_h, tg_v, tg_f, tg_hb );
        sync_gpu_threads();
        stats.time_B += (clock() - g_start) / CLOCKS_PER_SEC;
        tensor::copy( tc_h_g, tg_h );
        stats.add_sample( tc_h, tc_h_g );
    } 
    printf("\n");
    stats.print();

    tensor::free_space( tg_v );
    tensor::free_space( tg_hb );
    tensor::free_space( tg_h );
    tensor::free_space( tg_f );
    tensor::free_space( tc_v );
    tensor::free_space( tc_hb );
    tensor::free_space( tc_h );
    tensor::free_space( tc_h_g );
    tensor::free_space( tc_f );        
}

void test_conv2_full( int num_iter ){
    TTensor1D tg_vb ( V_MAX );
    TTensor3D tg_v  ( V_MAX, V_Y_MAX, V_X_MAX );
    TTensor3D tg_h  ( H_MAX, H_Y_MAX, H_X_MAX );
    TTensor4D tg_f  ( V_MAX, H_MAX,  F_Y_MAX, F_X_MAX );
 
	CTensor1D tc_vb ( V_MAX );
    CTensor3D tc_v  ( V_MAX, V_Y_MAX, V_X_MAX );
    CTensor3D tc_h  ( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor3D tc_v_g( V_MAX, V_Y_MAX, V_X_MAX );
    CTensor4D tc_f  ( V_MAX, H_MAX,  F_Y_MAX, F_X_MAX );

	tensor::alloc_space( tg_v );
    tensor::alloc_space( tg_vb );
    tensor::alloc_space( tg_h );
    tensor::alloc_space( tg_f );
    tensor::alloc_space( tc_v );
    tensor::alloc_space( tc_vb );
    tensor::alloc_space( tc_h );
    tensor::alloc_space( tc_v_g );
    tensor::alloc_space( tc_f );
    printf("start test conv2_full\n");
    
	TestStats<CTensor3D> stats( "conv2_full_CPU","conv2_full_GPU");
    stats.abs_err.set_param( H_MAX, H_Y_MAX, H_X_MAX );
    stats.abs_err_rel.set_param( H_MAX, H_Y_MAX, H_X_MAX );
    stats.abs_err_relT.set_param( H_MAX, H_Y_MAX, H_X_MAX );
    stats.init();
    
    for( int i = 0 ; i < num_iter ; i ++ ){
		printf("\r                                  \r");
		printf("round [%8d]", i);
        fflush( stdout );

		tensor::sample_gaussian( tc_f, sd ); 
		tensor::sample_gaussian( tc_h, sd );
		tensor::sample_gaussian( tc_vb, sd );

        double c_start = clock();
        tensor::crbm::conv2_full( tc_v, tc_h, tc_f, tc_vb );
        stats.time_A += (clock() - c_start) / CLOCKS_PER_SEC;

        tensor::copy( tg_f, tc_f );
        tensor::copy( tg_h, tc_h );        
        tensor::copy( tg_vb, tc_vb );
        
        double g_start = clock();
        tensor::crbm::conv2_full( tg_v, tg_h, tg_f, tg_vb );
        sync_gpu_threads();
        stats.time_B += (clock() - g_start) / CLOCKS_PER_SEC;
        tensor::copy( tc_v_g, tg_v );
        stats.add_sample( tc_v, tc_v_g );
    } 
    printf("\n");
    stats.print();

    tensor::free_space( tg_v );
    tensor::free_space( tg_vb );
    tensor::free_space( tg_h );
    tensor::free_space( tg_f );
    tensor::free_space( tc_vb );
    tensor::free_space( tc_h );
    tensor::free_space( tc_h );
    tensor::free_space( tc_v_g );
    tensor::free_space( tc_f );        
}

void test_conv2_r_big_filter( int num_iter ){
    TTensor3D tg_v  ( V_MAX, V_Y_MAX, V_X_MAX );
    TTensor3D tg_h  ( H_MAX, H_Y_MAX, H_X_MAX );
    TTensor4D tg_f  ( V_MAX, H_MAX,  F_Y_MAX, F_X_MAX );
    CTensor3D tc_v  ( V_MAX, V_Y_MAX, V_X_MAX );
    CTensor3D tc_h  ( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor4D tc_f_g( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );
    CTensor4D tc_f  ( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );

	tensor::alloc_space( tg_v );
    tensor::alloc_space( tg_h );
    tensor::alloc_space( tg_f );
    tensor::alloc_space( tc_v );
    tensor::alloc_space( tc_h );
    tensor::alloc_space( tc_f_g );
    tensor::alloc_space( tc_f );
    printf("start test conv2_r_big_filter\n");
    
	TestStats<CTensor4D> stats( "conv2_r_big_filter_CPU","conv2_r_big_filter_GPU");
    stats.abs_err.set_param( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );
    stats.abs_err_rel.set_param ( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );
    stats.abs_err_relT.set_param( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );
    stats.init();
    
    for( int i = 0 ; i < num_iter ; i ++ ){
		printf("\r                                  \r");
		printf("round [%8d]", i);
        fflush( stdout );

		tensor::sample_gaussian( tc_f, sd ); 
		tensor::sample_gaussian( tc_h, sd );
		tensor::sample_gaussian( tc_v, sd );
        
        tensor::copy( tg_f, tc_f );
        tensor::copy( tg_h, tc_h );        
        tensor::copy( tg_v, tc_v );

        double c_start = clock();
        tensor::crbm::sadd__conv2_r_big_filter( tc_f, tc_v, tc_h );
        stats.time_A += (clock() - c_start) / CLOCKS_PER_SEC;
        double g_start = clock();
        tensor::crbm::sadd__conv2_r_big_filter( tg_f, tg_v, tg_h );
        sync_gpu_threads();
        stats.time_B += (clock() - g_start) / CLOCKS_PER_SEC;
        tensor::copy( tc_f_g, tg_f );
        stats.add_sample( tc_f, tc_f_g );
    } 
    printf("\n");
    stats.print();

    tensor::free_space( tg_v );
    tensor::free_space( tg_h );
    tensor::free_space( tg_f );
    tensor::free_space( tc_h );
    tensor::free_space( tc_v );
    tensor::free_space( tc_f_g );
    tensor::free_space( tc_f );        
}


void test_norm_maxpooling_2D( int num_iter ){
    TTensor3D tg_hp ( H_MAX, H_Y_MAX, H_X_MAX );
    TTensor3D tg_h  ( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor3D tc_h     ( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor3D tc_hp    ( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor3D tc_hp_g  ( H_MAX, H_Y_MAX, H_X_MAX );


	tensor::alloc_space( tg_hp );
    tensor::alloc_space( tg_h );
    tensor::alloc_space( tc_hp );
    tensor::alloc_space( tc_h );
    tensor::alloc_space( tc_hp_g );
    printf("start test norm_maxpooling_2D\n");
    
	TestStats<CTensor3D> stats( "norm_maxpooling_2D_CPU","norm_maxpooling_2D_GPU");
    stats.abs_err.set_param( H_MAX, H_Y_MAX, H_X_MAX );
    stats.abs_err_rel.set_param( H_MAX, H_Y_MAX, H_X_MAX );
    stats.abs_err_relT.set_param( H_MAX, H_Y_MAX, H_X_MAX );
    stats.init();
    
    for( int i = 0 ; i < num_iter ; i ++ ){
		printf("\r                                  \r");
		printf("round [%8d]", i);
        fflush( stdout );

		tensor::sample_gaussian( tc_h, sd );
        
        tensor::copy( tg_h, tc_h );        

        double c_start = clock();
        tensor::crbm::norm_maxpooling_2D( tc_hp, tc_h, POOL_SIZE );
        tc_hp += 1.0f;
        tc_hp = tc_hp * 3.1f + tc_h * 0.001f; 
        tc_hp = tc_hp*tc_h;
        stats.time_A += (clock() - c_start) / CLOCKS_PER_SEC;
        double g_start = clock();
        tensor::crbm::norm_maxpooling_2D( tg_hp, tg_h, POOL_SIZE );
        tg_hp += 1.0f;
        tg_hp  = tg_hp * 3.1f + tg_h * 0.001f; 
        tg_hp  = tg_hp*tg_h;
        sync_gpu_threads();
        stats.time_B += (clock() - g_start) / CLOCKS_PER_SEC;
        tensor::copy( tc_hp_g, tg_hp );
        stats.add_sample( tc_hp, tc_hp_g );
    } 
    printf("\n");
    stats.print();

    tensor::free_space( tg_hp );
    tensor::free_space( tg_h );
    tensor::free_space( tc_hp );
    tensor::free_space( tc_h );
    tensor::free_space( tc_hp_g );
}

void test_pool_up( int num_iter ){
    TTensor3D tg_hp ( H_MAX, P_Y_MAX, P_X_MAX );
    TTensor3D tg_h  ( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor3D tc_h     ( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor3D tc_hp    ( H_MAX, P_Y_MAX, P_X_MAX );
    CTensor3D tc_hp_g  ( H_MAX, P_Y_MAX, P_X_MAX );


	tensor::alloc_space( tg_hp );
    tensor::alloc_space( tg_h );
    tensor::alloc_space( tc_hp );
    tensor::alloc_space( tc_h );
    tensor::alloc_space( tc_hp_g );
    printf("start test pool up\n");
    
	TestStats<CTensor3D> stats( "pool_up_CPU","pool_up_GPU");
    stats.abs_err.set_param( H_MAX, P_Y_MAX, P_X_MAX );
    stats.abs_err_rel.set_param( H_MAX, P_Y_MAX, P_X_MAX );
    stats.abs_err_relT.set_param( H_MAX, P_Y_MAX, P_X_MAX );
    stats.init();
    
    for( int i = 0 ; i < num_iter ; i ++ ){
		printf("\r                                  \r");
		printf("round [%8d]", i);
        fflush( stdout );

		tensor::sample_gaussian( tc_h, sd );
        
        tensor::copy( tg_h, tc_h );        

        double c_start = clock();
        tensor::crbm::pool_up( tc_hp, tc_h, POOL_SIZE );
        stats.time_A += (clock() - c_start) / CLOCKS_PER_SEC;
        double g_start = clock();
        tensor::crbm::pool_up( tg_hp, tg_h, POOL_SIZE );
        sync_gpu_threads();
        stats.time_B += (clock() - g_start) / CLOCKS_PER_SEC;
        tensor::copy( tc_hp_g, tg_hp );
        stats.add_sample( tc_hp, tc_hp_g );
    } 
    printf("\n");
    stats.print();

    tensor::free_space( tg_hp );
    tensor::free_space( tg_h );
    tensor::free_space( tc_hp );
    tensor::free_space( tc_h );
    tensor::free_space( tc_hp_g );
}

void test_sum_2D( int num_iter ){
    TTensor1D tg_s   ( H_MAX );
    TTensor3D tg_h   ( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor3D tc_h   ( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor1D tc_s   ( H_MAX );
    CTensor1D tc_s_g( H_MAX );

	tensor::alloc_space( tg_s );
    tensor::alloc_space( tg_h );
    tensor::alloc_space( tc_s );
    tensor::alloc_space( tc_h );
    tensor::alloc_space( tc_s_g );
    printf("start test sum2D\n");
    
	TestStats<CTensor1D> stats( "sum2D_CPU","sum2D_GPU");
    stats.abs_err.set_param( H_MAX );
    stats.abs_err_rel.set_param( H_MAX );
    stats.abs_err_relT.set_param( H_MAX );
    stats.init();
    
    for( int i = 0 ; i < num_iter ; i ++ ){
		printf("\r                                  \r");
		printf("round [%8d]", i);
        fflush( stdout );

        tensor::sample_gaussian( tc_h, sd );
        tensor::sample_gaussian( tc_s, sd );
        
        tensor::copy( tg_h, tc_h );        
        tensor::copy( tg_s, tc_s );        
        
        double c_start = clock();
        if( (i&1) == 0 ) 
            tc_s += sum_2D( tc_h );
        else
            tc_s -= sum_2D( tc_h );

        stats.time_A += (clock() - c_start) / CLOCKS_PER_SEC;
        double g_start = clock();
        
        if( (i&1) == 0 ) 
            tg_s += sum_2D( tg_h );
        else
            tg_s -= sum_2D( tg_h );

        sync_gpu_threads();
        stats.time_B += (clock() - g_start) / CLOCKS_PER_SEC;
        tensor::copy( tc_s_g, tg_s );
        stats.add_sample( tc_s, tc_s_g );
    } 
    printf("\n");
    stats.print();

    tensor::free_space( tg_s );
    tensor::free_space( tg_h );
    tensor::free_space( tc_s );
    tensor::free_space( tc_h );
    tensor::free_space( tc_s_g );
}

void test_sum_2DX( int num_iter ){
    TTensor2D tg_s   ( V_MAX, H_MAX );
    TTensor4D tg_h   ( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );
    CTensor4D tc_h   ( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );
    CTensor2D tc_s   ( V_MAX, H_MAX );
    CTensor2D tc_s_g ( V_MAX, H_MAX );

	tensor::alloc_space( tg_s );
    tensor::alloc_space( tg_h );
    tensor::alloc_space( tc_s );
    tensor::alloc_space( tc_h );
    tensor::alloc_space( tc_s_g );
    printf("start test sum2DX\n");
    
	TestStats<CTensor2D> stats( "sum2DX_CPU","sum2DX_GPU");
    stats.abs_err.set_param( V_MAX, H_MAX );
    stats.abs_err_rel.set_param( V_MAX, H_MAX );
    stats.abs_err_relT.set_param( V_MAX, H_MAX );
    stats.init();
    
    for( int i = 0 ; i < num_iter ; i ++ ){
		printf("\r                                  \r");
		printf("round [%8d]", i);
        fflush( stdout );
        
		tensor::sample_gaussian( tc_h, sd );
        tensor::sample_gaussian( tc_s, sd );
        
        tensor::copy( tg_h, tc_h );        
        tensor::copy( tg_s, tc_s );        
        
        double c_start = clock();

        tensor::crbm::sum_2D( tc_s, tc_h );

        stats.time_A += (clock() - c_start) / CLOCKS_PER_SEC;
        double g_start = clock();
        
        tensor::crbm::sum_2D( tg_s, tg_h );

        sync_gpu_threads();
        stats.time_B += (clock() - g_start) / CLOCKS_PER_SEC;
        tensor::copy( tc_s_g, tg_s );
        stats.add_sample( tc_s, tc_s_g );
    } 
    printf("\n");
    stats.print();

    tensor::free_space( tg_s );
    tensor::free_space( tg_h );
    tensor::free_space( tc_s );
    tensor::free_space( tc_h );
    tensor::free_space( tc_s_g );
}


void test_sadd__scale( int num_iter ){
    TTensor2D tg_s   ( V_MAX, H_MAX );
    TTensor4D tg_h   ( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );
    CTensor4D tc_h   ( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );
    CTensor2D tc_s   ( V_MAX, H_MAX );
    CTensor4D tc_h_g ( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );

	tensor::alloc_space( tg_s );
    tensor::alloc_space( tg_h );
    tensor::alloc_space( tc_s );
    tensor::alloc_space( tc_h );
    tensor::alloc_space( tc_h_g );
    printf("start sadd__scale\n");
    
	TestStats<CTensor4D> stats( "sadd__scale_CPU","sdd__scale_GPU");
    stats.abs_err.set_param( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );
    stats.abs_err_rel.set_param( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );
    stats.abs_err_relT.set_param( V_MAX, H_MAX, F_Y_MAX, F_X_MAX );
    stats.init();
    
    for( int i = 0 ; i < num_iter ; i ++ ){
		printf("\r                                  \r");
		printf("round [%8d]", i);
        fflush( stdout );

		tensor::sample_gaussian( tc_h, sd );
        tensor::sample_gaussian( tc_s, sd );

        tensor::copy( tg_h, tc_h );        
        tensor::copy( tg_s, tc_s );        
        
        double c_start = clock();

        tensor::crbm::sadd__scale( tc_h, tc_s, 0.2f );

        stats.time_A += (clock() - c_start) / CLOCKS_PER_SEC;
        double g_start = clock();
        
        tensor::crbm::sadd__scale( tg_h, tg_s, 0.2f );

        sync_gpu_threads();
        stats.time_B += (clock() - g_start) / CLOCKS_PER_SEC;
        tensor::copy( tc_h_g, tg_h );
        stats.add_sample( tc_h, tc_h_g );
    } 
    printf("\n");
    stats.print();

    tensor::free_space( tg_s );
    tensor::free_space( tg_h );
    tensor::free_space( tc_s );
    tensor::free_space( tc_h );
    tensor::free_space( tc_h_g );
}

void test_add_sparse_info( int num_iter ){
    TTensor1D tg_h_mf     ( H_MAX );
    TTensor1D tg_h_mf_grad( H_MAX );
    TTensor3D tg_h   ( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor3D tc_h   ( H_MAX, H_Y_MAX, H_X_MAX );
    CTensor1D tc_h_mf( H_MAX );
    CTensor1D tc_h_mf_grad( H_MAX );
    CTensor1D tc_h_mf_g( H_MAX );
    CTensor1D tc_h_mf_grad_g( H_MAX );

	tensor::alloc_space( tg_h_mf );
    tensor::alloc_space( tg_h_mf_grad );
    tensor::alloc_space( tg_h );
    tensor::alloc_space( tc_h_mf );
    tensor::alloc_space( tc_h_mf_grad );
    tensor::alloc_space( tc_h );
    tensor::alloc_space( tc_h_mf_g );
    tensor::alloc_space( tc_h_mf_grad_g );

    printf("start test add_sparse_info\n");
    
	TestStats<CTensor1D> stats( "add_sparse_info_CPU","add_sparse_info_GPU");
    stats.abs_err.set_param( H_MAX );
    stats.abs_err_rel.set_param( H_MAX );
    stats.abs_err_relT.set_param( H_MAX );
    stats.init();
    
    for( int i = 0 ; i < num_iter ; i ++ ){
		printf("\r                                  \r");
		printf("round [%8d]", i);
        fflush( stdout );

		tensor::sample_gaussian( tc_h, sd );
        tensor::sample_gaussian( tc_h_mf, sd );
        tensor::sample_gaussian( tc_h_mf_grad, sd );
        
        tensor::copy( tg_h, tc_h );        
        tensor::copy( tg_h_mf, tc_h_mf );        
        tensor::copy( tg_h_mf_grad, tc_h_mf_grad );        
        
        double c_start = clock();
        tensor::crbm::add_sparse_info( tc_h_mf, tc_h_mf_grad, tc_h, POOL_SIZE );

        stats.time_A += (clock() - c_start) / CLOCKS_PER_SEC;
        double g_start = clock();
        tensor::crbm::add_sparse_info( tg_h_mf, tg_h_mf_grad, tg_h, POOL_SIZE );
        sync_gpu_threads();
        stats.time_B += (clock() - g_start) / CLOCKS_PER_SEC;
        tensor::copy( tc_h_mf_g, tg_h_mf );
        tensor::copy( tc_h_mf_grad_g, tg_h_mf_grad );
        stats.add_sample( tc_h_mf, tc_h_mf_g );
        stats.add_sample( tc_h_mf_grad, tc_h_mf_grad_g );
    } 
    printf("\n");
    stats.print();

    tensor::free_space( tg_h_mf );
    tensor::free_space( tg_h_mf_grad );
    tensor::free_space( tg_h );
    tensor::free_space( tc_h_mf );
    tensor::free_space( tc_h_mf_grad );
    tensor::free_space( tc_h );
    tensor::free_space( tc_h_mf_g );
    tensor::free_space( tc_h_mf_grad_g );
}

void test_refill_edge_area( int num_iter ){
    TTensor3D tg_vp ( V_MAX, V_Y_MAX, V_X_MAX );
    TTensor3D tg_v  ( V_MAX, V_Y_MAX, V_X_MAX );
    CTensor3D tc_v     ( V_MAX, V_Y_MAX, V_X_MAX );
    CTensor3D tc_vp    ( V_MAX, V_Y_MAX, V_X_MAX );
    CTensor3D tc_vp_g  ( V_MAX, V_Y_MAX, V_X_MAX );

	tensor::alloc_space( tg_vp );
    tensor::alloc_space( tg_v );
    tensor::alloc_space( tc_vp );
    tensor::alloc_space( tc_v );
    tensor::alloc_space( tc_vp_g );
    printf("start test refill_edge_area\n");
    
	TestStats<CTensor3D> stats( "refill_edge_area_CPU","refill_edge_area_GPU");
    stats.abs_err.set_param( V_MAX, V_Y_MAX, V_X_MAX );
    stats.abs_err_rel.set_param( V_MAX, V_Y_MAX, V_X_MAX );
    stats.abs_err_relT.set_param( V_MAX, V_Y_MAX, V_X_MAX );
    stats.init();
    
    for( int i = 0 ; i < num_iter ; i ++ ){                
        printf("\r                                           \r");
		printf("round [%8d]", i);
        fflush( stdout );
		
        tensor::sample_gaussian( tc_v, sd );
        tensor::sample_gaussian( tc_vp, sd );
        
        tensor::copy( tg_v, tc_v );        
        tensor::copy( tg_vp, tc_vp );        

        double c_start = clock();
        tensor::crbm::refill_edge_area( tc_vp, tc_v, F_Y_MAX-1, F_X_MAX-1 );
        stats.time_A += (clock() - c_start) / CLOCKS_PER_SEC;
        double g_start = clock();
        tensor::crbm::refill_edge_area( tg_vp, tg_v, F_Y_MAX-1, F_X_MAX-1 );
        sync_gpu_threads();
        stats.time_B += (clock() - g_start) / CLOCKS_PER_SEC;
        tensor::copy( tc_vp_g, tg_vp );
        stats.add_sample( tc_vp, tc_vp_g );
    } 
    printf("\n");
    stats.print();

    tensor::free_space( tg_vp );
    tensor::free_space( tg_v );
    tensor::free_space( tc_vp );
    tensor::free_space( tc_v );
    tensor::free_space( tc_vp_g );
}

#endif

