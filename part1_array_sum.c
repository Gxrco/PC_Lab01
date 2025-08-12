#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ARRAY_SIZE 100000000

int main() {
    int *array;
    long long sequential_sum = 0;
    long long parallel_sum_no_reduction = 0;
    long long parallel_sum_with_reduction = 0;
    double start_time, end_time;
    double sequential_time, parallel_time_no_reduction, parallel_time_with_reduction;
    
    // Allocate memory for array
    array = (int *)malloc(ARRAY_SIZE * sizeof(int));
    if (array == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    
    // Initialize array with values 0 to 99,999,999
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = i;
    }
    
    printf("Array size: %d elements\n", ARRAY_SIZE);
    printf("Number of threads available: %d\n", omp_get_max_threads());
    printf("\n");
    
    // SEQUENTIAL VERSION
    printf("=== SEQUENTIAL VERSION ===\n");
    start_time = omp_get_wtime();
    
    for (int i = 0; i < ARRAY_SIZE; i++) {
        sequential_sum += array[i];
    }
    
    end_time = omp_get_wtime();
    sequential_time = end_time - start_time;
    
    printf("Sequential sum: %lld\n", sequential_sum);
    printf("Sequential time: %f seconds\n", sequential_time);
    printf("\n");
    
    // PARALLEL VERSION WITHOUT REDUCTION
    printf("=== PARALLEL VERSION WITHOUT REDUCTION ===\n");
    start_time = omp_get_wtime();
    
    #pragma omp parallel
    {
        long long local_sum = 0;
        #pragma omp for
        for (int i = 0; i < ARRAY_SIZE; i++) {
            local_sum += array[i];
        }
        
        #pragma omp critical
        {
            parallel_sum_no_reduction += local_sum;
        }
    }
    
    end_time = omp_get_wtime();
    parallel_time_no_reduction = end_time - start_time;
    
    printf("Parallel sum (no reduction): %lld\n", parallel_sum_no_reduction);
    printf("Parallel time (no reduction): %f seconds\n", parallel_time_no_reduction);
    printf("\n");
    
    // PARALLEL VERSION WITH REDUCTION
    printf("=== PARALLEL VERSION WITH REDUCTION ===\n");
    start_time = omp_get_wtime();
    
    #pragma omp parallel for reduction(+:parallel_sum_with_reduction)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        parallel_sum_with_reduction += array[i];
    }
    
    end_time = omp_get_wtime();
    parallel_time_with_reduction = end_time - start_time;
    
    printf("Parallel sum (with reduction): %lld\n", parallel_sum_with_reduction);
    printf("Parallel time (with reduction): %f seconds\n", parallel_time_with_reduction);
    printf("\n");
    
    // RESULTS ANALYSIS
    printf("=== PERFORMANCE ANALYSIS ===\n");
    printf("Sequential time: %f seconds\n", sequential_time);
    printf("Parallel time (no reduction): %f seconds\n", parallel_time_no_reduction);
    printf("Parallel time (with reduction): %f seconds\n", parallel_time_with_reduction);
    printf("\n");
    
    printf("Speedup (no reduction): %.2fx\n", sequential_time / parallel_time_no_reduction);
    printf("Speedup (with reduction): %.2fx\n", sequential_time / parallel_time_with_reduction);
    printf("\n");
    
    printf("Efficiency (no reduction): %.2f%%\n", (sequential_time / parallel_time_no_reduction) / omp_get_max_threads() * 100);
    printf("Efficiency (with reduction): %.2f%%\n", (sequential_time / parallel_time_with_reduction) / omp_get_max_threads() * 100);
    printf("\n");
    
    // Verify correctness
    if (sequential_sum == parallel_sum_no_reduction && sequential_sum == parallel_sum_with_reduction) {
        printf("✓ All results are correct and consistent\n");
    } else {
        printf("✗ Results are inconsistent!\n");
        printf("Sequential: %lld, Parallel (no reduction): %lld, Parallel (with reduction): %lld\n",
               sequential_sum, parallel_sum_no_reduction, parallel_sum_with_reduction);
    }
    
    free(array);
    return 0;
}