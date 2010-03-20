#ifndef _TEST_CMP_H_
#define _TEST_CMP_H_

#include "../apex_tensor.h"
#include "test_stats.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
using namespace apex_tensor;

void test_conv2_r_valid( int num_iter ){
    GTensor1D tg_hb ( H_MAX );
    GTensor3D tg_v  ( V_MAX, V_Y_MAX, V_X_MAX );
    GTensor3D tg_h  ( H_MAX, H_Y_MAX, H_X_MAX );
    GTensor4D tg_f  ( V_MAX, H_MAX,  F_Y_MAX, F_X_MAX );
 
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
    GTensor1D tg_vb ( V_MAX );
    GTensor3D tg_v  ( V_MAX, V_Y_MAX, V_X_MAX );
    GTensor3D tg_h  ( H_MAX, H_Y_MAX, H_X_MAX );
    GTensor4D tg_f  ( V_MAX, H_MAX,  F_Y_MAX, F_X_MAX );
 
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
    GTensor3D tg_v  ( V_MAX, V_Y_MAX, V_X_MAX );
    GTensor3D tg_h  ( H_MAX, H_Y_MAX, H_X_MAX );
    GTensor4D tg_f  ( V_MAX, H_MAX,  F_Y_MAX, F_X_MAX );
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

#endif

