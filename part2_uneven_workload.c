#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifndef N
#define N 5000                 
#endif
#define BASE_WORK 40            
#define WORK_DIVISOR 2500       
#define PRINT_FIRST 20         

static inline double simulate_work(int iteration) {
    int work_amount = BASE_WORK + (int)((1.0 * iteration * iteration) / WORK_DIVISOR);

    volatile double acc = 0.0; 
    for (int k = 0; k < work_amount; ++k) {
        acc += (iteration * 1315423911u + k * 2654435761u) * 1e-12;
        acc -= (k & 7) * 1e-12;
    }
    return acc;
}

static void run_static(int chunk_size) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    if (chunk_size > 0) {
        #pragma omp parallel for schedule(static, chunk_size) reduction(+:total_result)
        for (int i = 0; i < N; i++) {
            total_result += simulate_work(i);
            if (i < PRINT_FIRST) {
                #pragma omp critical
                {
                    printf("[static,%d] thread %d -> i=%d\n", chunk_size, omp_get_thread_num(), i);
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static) reduction(+:total_result)
        for (int i = 0; i < N; i++) {
            total_result += simulate_work(i);
            if (i < PRINT_FIRST) {
                #pragma omp critical
                {
                    printf("[static,default] thread %d -> i=%d\n", omp_get_thread_num(), i);
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    printf("Execution time (static%s%d): %.6f s\n",
           chunk_size>0?", ":"", chunk_size>0?chunk_size:0, end_time - start_time);
    printf("Total result: %.6f\n", total_result);
}

static void run_dynamic(int chunk_size) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    #pragma omp parallel for schedule(dynamic, chunk_size) reduction(+:total_result)
    for (int i = 0; i < N; i++) {
        total_result += simulate_work(i);
        if (i < PRINT_FIRST) {
            #pragma omp critical
            {
                printf("[dynamic,%d] thread %d -> i=%d\n", chunk_size, omp_get_thread_num(), i);
            }
        }
    }

    double end_time = omp_get_wtime();
    printf("Execution time (dynamic, %d): %.6f s\n", chunk_size, end_time - start_time);
    printf("Total result: %.6f\n", total_result);
}

static void run_guided(int chunk_size) {
    double start_time = omp_get_wtime();
    double total_result = 0.0;

    #pragma omp parallel for schedule(guided, chunk_size) reduction(+:total_result)
    for (int i = 0; i < N; i++) {
        total_result += simulate_work(i);
        if (i < PRINT_FIRST) {
            #pragma omp critical
            {
                printf("[guided,%d] thread %d -> i=%d\n", chunk_size, omp_get_thread_num(), i);
            }
        }
    }

    double end_time = omp_get_wtime();
    printf("Execution time (guided, %d): %.6f s\n", chunk_size, end_time - start_time);
    printf("Total result: %.6f\n", total_result);
}

int main() {
    printf("Uneven Workload Simulation (FAST)\n");
    printf("N = %d iterations\n", N);
    printf("Threads available: %d\n", omp_get_max_threads());
    printf("Work ~ i^2 scaled (BASE=%d, DIV=%d)\n\n", BASE_WORK, WORK_DIVISOR);

    int static_chunks[]  = {0, 256};
    int dynamic_chunks[] = {1, 64};
    int guided_chunks[]  = {1, 64};

    printf("===== STATIC =====\n");
    for (size_t i = 0; i < sizeof(static_chunks)/sizeof(static_chunks[0]); ++i) {
        run_static(static_chunks[i]);
    }

    printf("\n===== DYNAMIC =====\n");
    for (size_t i = 0; i < sizeof(dynamic_chunks)/sizeof(dynamic_chunks[0]); ++i) {
        run_dynamic(dynamic_chunks[i]);
    }

    printf("\n===== GUIDED =====\n");
    for (size_t i = 0; i < sizeof(guided_chunks)/sizeof(guided_chunks[0]); ++i) {
        run_guided(guided_chunks[i]);
    }

    // Sequential baseline for reference
    printf("\n===== SEQUENTIAL (baseline) =====\n");
    double seq_start = omp_get_wtime();
    double seq_total = 0.0;
    for (int i = 0; i < N; ++i) seq_total += simulate_work(i);
    double seq_end = omp_get_wtime();
    printf("Sequential time: %.6f s\n", seq_end - seq_start);
    printf("Sequential result: %.6f\n", seq_total);

    return 0;
}
