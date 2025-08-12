#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE 50000000
#endif
#ifndef TRIALS
#define TRIALS 3                 // run each config multiple times; keep the best time
#endif

static int *g_array = NULL;
static const int MODVAL = 1000;   // values are i % MODVAL

static long long expected_sum(long long n) {
    // Sum_{i=0..n-1} (i % MODVAL)
    long long cycle_sum = (long long)(MODVAL - 1) * MODVAL / 2; // 0..999 = 499500
    long long cycles = n / MODVAL;
    long long rem = n % MODVAL;
    long long rem_sum = rem * (rem - 1) / 2;
    return cycles * cycle_sum + rem_sum;
}

static void init_array_once() {
    if (!g_array) {
        g_array = (int *)malloc(sizeof(int) * (size_t)ARRAY_SIZE);
        if (!g_array) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
        for (int i = 0; i < ARRAY_SIZE; ++i) g_array[i] = i % MODVAL;
    }
}

static double run_sequential(long long *sum_out) {
    long long sum = 0;
    double best = 1e100;
    for (int t = 0; t < TRIALS; ++t) {
        double t0 = omp_get_wtime();
        for (int i = 0; i < ARRAY_SIZE; ++i) sum += g_array[i];
        double t1 = omp_get_wtime();
        if (t1 - t0 < best) best = t1 - t0;
        sum = 0; // reset between trials to be fair
    }
    // one final pass to get the displayed sum
    for (int i = 0; i < ARRAY_SIZE; ++i) sum += g_array[i];
    if (sum_out) *sum_out = sum;
    return best;
}

static double run_parallel(int requested_threads, int *actual_threads_out, long long *sum_out) {
    // Fix the team size; avoid OpenMP dynamically changing it between runs
    omp_set_dynamic(0);
    omp_set_num_threads(requested_threads);

    long long sum = 0;
    int actual_threads = 0;
    double best = 1e100;

    for (int t = 0; t < TRIALS; ++t) {
        sum = 0;
        double t0 = omp_get_wtime();
        #pragma omp parallel
        {
            if (omp_get_thread_num() == 0) actual_threads = omp_get_num_threads();
            #pragma omp for reduction(+:sum) schedule(static)
            for (int i = 0; i < ARRAY_SIZE; ++i) {
                sum += g_array[i];
            }
        }
        double t1 = omp_get_wtime();
        if (t1 - t0 < best) best = t1 - t0;
    }

    if (actual_threads_out) *actual_threads_out = actual_threads;
    if (sum_out) *sum_out = sum;
    return best;
}

static int contains(const int *arr, int n, int v) {
    for (int i = 0; i < n; ++i) if (arr[i] == v) return 1; return 0;
}

int main() {
    printf("Performance Analysis - Scalability Study (Improved)\n");
    init_array_once();

    const int MAX_THREADS_INIT = omp_get_max_threads(); // capture once; don't let later calls affect logic
    printf("Array size: %d elements\n", ARRAY_SIZE);
    printf("Maximum threads available (initial): %d\n\n", MAX_THREADS_INIT);

    // Build test set: {1,2,4,8,16,32} intersected with [1..MAX_THREADS_INIT], plus MAX_THREADS_INIT if missing
    int candidates[] = {1,2,4,8,16,32};
    int tests[16];
    int nt = 0;
    for (int i = 0; i < (int)(sizeof(candidates)/sizeof(candidates[0])); ++i) {
        if (candidates[i] <= MAX_THREADS_INIT && !contains(tests, nt, candidates[i])) tests[nt++] = candidates[i];
    }
    if (!contains(tests, nt, MAX_THREADS_INIT)) tests[nt++] = MAX_THREADS_INIT; // ensure we test the real max (e.g., 11)

    // Sequential baseline
    long long seq_sum = 0;
    double seq_time = run_sequential(&seq_sum);
    long long exp_sum = expected_sum(ARRAY_SIZE);
    printf("=== SEQUENTIAL BASELINE ===\n");
    printf("Sequential - Sum: %lld (expected %lld), Best Time over %d run(s): %.6f s\n\n", seq_sum, exp_sum, TRIALS, seq_time);

    // Parallel tests
    printf("=== PARALLEL PERFORMANCE TESTING ===\n");
    double times[16];
    int actual_threads[16];
    long long par_sum = 0;

    for (int i = 0; i < nt; ++i) {
        times[i] = run_parallel(tests[i], &actual_threads[i], &par_sum);
        printf("Requested: %2d, Actual: %2d, Sum: %lld, Best Time: %.6f s\n", tests[i], actual_threads[i], par_sum, times[i]);
    }

    // Table
    printf("\n=== PERFORMANCE ANALYSIS TABLE ===\n");
    printf("%-8s %-12s %-10s %-12s %-12s\n", "Threads", "Time (sec)", "Speedup", "Efficiency", "Notes");
    printf("------------------------------------------------------------\n");

    double best_speedup = 0.0; int best_threads = tests[0];
    for (int i = 0; i < nt; ++i) {
        double speedup = seq_time / times[i];
        double efficiency = (speedup / actual_threads[i]) * 100.0;
        if (speedup > best_speedup) { best_speedup = speedup; best_threads = actual_threads[i]; }
        const char *note = (efficiency > 80.0) ? "Excellent" : (efficiency > 60.0) ? "Good" : (efficiency > 40.0) ? "Fair" : "Poor";
        printf("%-8d %-12.4f %-10.2f %-11.2f%% %s\n", actual_threads[i], times[i], speedup, efficiency, note);
    }

    printf("\n=== OPTIMAL CONFIGURATION ===\n");
    printf("Best performance: %d threads\n", best_threads);
    printf("Maximum speedup: %.2fx\n", best_speedup);
    printf("Best efficiency: %.2f%%\n\n", (best_speedup / best_threads) * 100.0);

    // Trend analysis (monotonicity check)
    printf("=== TREND ANALYSIS ===\n");
    for (int i = 1; i < nt; ++i) {
        double s_prev = seq_time / times[i-1];
        double s_cur  = seq_time / times[i];
        if (s_cur + 1e-9 < 0.95 * s_prev) {
            printf("- Performance degradation detected at %d threads (%.2fx -> %.2fx)\n", actual_threads[i], s_prev, s_cur);
        }
    }

    // CSV output
    printf("=== CSV DATA FOR GRAPHING ===\n");
    printf("Threads,Time,Speedup,Efficiency\n");
    for (int i = 0; i < nt; ++i) {
        double speedup = seq_time / times[i];
        double efficiency = (speedup / actual_threads[i]) * 100.0;
        printf("%d,%.6f,%.2f,%.2f\n", actual_threads[i], times[i], speedup, efficiency);
    }

    free(g_array);
    return 0;
}
