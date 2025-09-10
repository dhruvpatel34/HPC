#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrix_vector_multiply(int **A, int x[], int y[], int n) {
    #pragma omp parallel for schedule(guided) 
    for (int i = 0; i < n; i++) {
        register int sum = 0; 
        for (int j = 0; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        y[i] = sum;
    }
}

// Better memory allocation - contiguous memory for better cache performance
int** allocate_matrix_optimized(int n) {
    int **matrix = malloc(n * sizeof(int*));
    int *data = malloc(n * n * sizeof(int));
    for (int i = 0; i < n; i++) {
        matrix[i] = data + i * n;
    }
    return matrix;
}

void free_matrix_optimized(int **matrix) {
    free(matrix[0]); // Free the data block
    free(matrix);    // Free the pointer array
}

void initialize_matrix(int **A, int *x, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = i + 1;
        for (int j = 0; j < n; j++) {
            A[i][j] = (i + j) % 10 + 1;
        }
    }
}

int main() {
    int sizes[] = {1, 10, 20, 50, 100, 200, 500, 1000, 1200, 1500};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    // Set number of threads explicitly
    omp_set_num_threads(4);
    int num_threads = omp_get_max_threads();
    
    printf("Using %d OpenMP threads\n", num_threads);
    printf("Matrix Size\tParallel Time (ms)\n");
    printf("---------------------------------\n");
    
    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        int **A = allocate_matrix_optimized(n);
        int *x = malloc(n * sizeof(int));
        int *y = malloc(n * sizeof(int));
        
        initialize_matrix(A, x, n);
        
        // Multiple runs for better timing accuracy
        double total_time = 0.0;
        int num_runs = (n < 100) ? 1000 : 10;
        
        for (int run = 0; run < num_runs; run++) {
            double start_time = omp_get_wtime();
            matrix_vector_multiply(A, x, y, n);
            double end_time = omp_get_wtime();
            total_time += (end_time - start_time);
        }
        
        double avg_time_ms = (total_time / num_runs) * 1000.0;
        printf("%d\t\t%.6f\n", n, avg_time_ms);
        
        free_matrix_optimized(A);
        free(x);
        free(y);
    }
    
    return 0;
}