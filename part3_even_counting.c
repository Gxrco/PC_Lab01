#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define ARRAY_SIZE 10000000

int main() {
    int *array;
    int sequential_count = 0;
    int parallel_count_critical = 0;
    int parallel_count_reduction = 0;
    double start_time, end_time;
    double sequential_time, critical_time, reduction_time;
    
    // Allocate memory for array
    array = (int *)malloc(ARRAY_SIZE * sizeof(int));
    if (array == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    
    // Initialize array with random integers
    srand(time(NULL));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = rand() % 1000; // Random numbers 0-999
    }
    
    printf("Even Number Counting - Race Condition Analysis\n");
    printf("Array size: %d elements\n", ARRAY_SIZE);
    printf("Number of threads: %d\n", omp_get_max_threads());
    printf("\n");
    
    // SEQUENTIAL VERSION
    printf("=== SEQUENTIAL VERSION ===\n");
    start_time = omp_get_wtime();
    
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (array[i] % 2 == 0) {
            sequential_count++;
        }
    }
    
    end_time = omp_get_wtime();
    sequential_time = end_time - start_time;
    
    printf("Sequential count: %d even numbers\n", sequential_count);
    printf("Sequential time: %f seconds\n", sequential_time);
    printf("\n");
    
    // PARALLEL VERSION WITH CRITICAL SECTION
    printf("=== PARALLEL VERSION WITH CRITICAL SECTION ===\n");
    start_time = omp_get_wtime();
    
    #pragma omp parallel for
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (array[i] % 2 == 0) {
            #pragma omp critical
            {
                parallel_count_critical++;
            }
        }
    }
    
    end_time = omp_get_wtime();
    critical_time = end_time - start_time;
    
    printf("Parallel count (critical): %d even numbers\n", parallel_count_critical);
    printf("Parallel time (critical): %f seconds\n", critical_time);
    printf("\n");
    
    // PARALLEL VERSION WITH REDUCTION
    printf("=== PARALLEL VERSION WITH REDUCTION ===\n");
    start_time = omp_get_wtime();
    
    #pragma omp parallel for reduction(+:parallel_count_reduction)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (array[i] % 2 == 0) {
            parallel_count_reduction++;
        }
    }
    
    end_time = omp_get_wtime();
    reduction_time = end_time - start_time;
    
    printf("Parallel count (reduction): %d even numbers\n", parallel_count_reduction);
    printf("Parallel time (reduction): %f seconds\n", reduction_time);
    printf("\n");
    
    // PERFORMANCE ANALYSIS
    printf("=== PERFORMANCE ANALYSIS ===\n");
    printf("Sequential time: %f seconds\n", sequential_time);
    printf("Critical section time: %f seconds\n", critical_time);
    printf("Reduction time: %f seconds\n", reduction_time);
    printf("\n");
    
    // Calculate speedup
    double critical_speedup = sequential_time / critical_time;
    double reduction_speedup = sequential_time / reduction_time;
    
    printf("Speedup (critical): %.2fx\n", critical_speedup);
    printf("Speedup (reduction): %.2fx\n", reduction_speedup);
    printf("\n");
    
    // Calculate efficiency
    int num_threads = omp_get_max_threads();
    printf("Efficiency (critical): %.2f%%\n", (critical_speedup / num_threads) * 100);
    printf("Efficiency (reduction): %.2f%%\n", (reduction_speedup / num_threads) * 100);
    printf("\n");
    
    // Performance comparison
    if (critical_time > reduction_time) {
        printf("Reduction is %.2fx faster than critical section\n", critical_time / reduction_time);
        printf("Slowest method: Critical section\n");
    } else {
        printf("Critical section is %.2fx faster than reduction\n", reduction_time / critical_time);
        printf("Slowest method: Reduction\n");
    }
    printf("\n");
    
    // Verify correctness
    if (sequential_count == parallel_count_critical && sequential_count == parallel_count_reduction) {
        printf("✓ All results are correct and consistent\n");
    } else {
        printf("✗ Results are inconsistent!\n");
        printf("Sequential: %d, Critical: %d, Reduction: %d\n",
               sequential_count, parallel_count_critical, parallel_count_reduction);
    }
    
    printf("\n=== ANALYSIS EXPLANATION ===\n");
    printf("Critical Section Approach:\n");
    printf("- Forces serialization at the increment operation\n");
    printf("- Creates contention as threads wait for the critical section\n");
    printf("- Overhead increases with number of threads\n");
    printf("- Good for complex operations that can't be reduced\n");
    printf("\n");
    
    printf("Reduction Approach:\n");
    printf("- Each thread maintains a private copy of the counter\n");
    printf("- Combines results efficiently at the end\n");
    printf("- Minimizes synchronization overhead\n");
    printf("- Optimal for associative operations like counting\n");
    printf("\n");
    
    // Demonstrate race condition (unsafe version)
    printf("=== RACE CONDITION DEMONSTRATION ===\n");
    printf("Running unsafe parallel version (with race condition)...\n");
    
    int unsafe_count = 0;
    start_time = omp_get_wtime();
    
    #pragma omp parallel for
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (array[i] % 2 == 0) {
            unsafe_count++; // Race condition here!
        }
    }
    
    end_time = omp_get_wtime();
    
    printf("Unsafe count: %d even numbers\n", unsafe_count);
    printf("Unsafe time: %f seconds\n", end_time - start_time);
    printf("Expected count: %d even numbers\n", sequential_count);
    
    if (unsafe_count != sequential_count) {
        printf("✗ Race condition detected! Count is incorrect.\n");
    } else {
        printf("⚠ Race condition may not have manifested in this run.\n");
    }
    
    free(array);
    return 0;
}