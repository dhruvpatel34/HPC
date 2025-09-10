#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrix_vector_multiply_2d_adaptive(int **A, int x[], int y[], int n) {
    // using sequential for very small matrices
    if (n <= 20) {
        for (int i = 0; i < n; i++) {
            register int sum = 0;
            for (int j = 0; j < n; j++) {
                sum += A[i][j] * x[j];
            }
            y[i] = sum;
        }
        return;
    }

    if (n <= 200) {
        #pragma omp parallel for schedule(static) num_threads(4)
        for (int i = 0; i < n; i++) {
            register int sum = 0;
            for (int j = 0; j < n; j++) {
                sum += A[i][j] * x[j];
            }
            y[i] = sum;
        }
        return;
    }
    
    const int BLOCK_SIZE = 64; 
    
    // Initialize result vector
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        y[i] = 0;
    }
    
   
    #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(4)
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
            int i_end = (ii + BLOCK_SIZE < n) ? ii + BLOCK_SIZE : n;
            int j_end = (jj + BLOCK_SIZE < n) ? jj + BLOCK_SIZE : n;
            
            for (int i = ii; i < i_end; i++) {
                register int local_sum = 0;
                for (int j = jj; j < j_end; j++) {
                    local_sum += A[i][j] * x[j];
                }
                #pragma omp atomic
                y[i] += local_sum;
            }
        }
    }
}

int** allocate_matrix_optimized(int n) {
    int **matrix = malloc(n * sizeof(int*));
    int *data = malloc(n * n * sizeof(int));
    for (int i = 0; i < n; i++) {
        matrix[i] = data + i * n;
    }
    return matrix;
}

void free_matrix_optimized(int **matrix) {
    free(matrix[0]);
    free(matrix);
}

void initialize_matrix(int **A, int *x, int n) {
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
    
    printf("Using 4 OpenMP threads 2D adaptive partition\n");
    printf("Matrix Size\tParallel Time (ms)\n");
    printf("---------------------------------\n");
    
    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        int **A = allocate_matrix_optimized(n);
        int *x = malloc(n * sizeof(int));
        int *y = malloc(n * sizeof(int));
        
        initialize_matrix(A, x, n);
        
        int num_runs = (n < 100) ? 1000 : 10;
        
        double total_time = 0.0;
        for (int run = 0; run < num_runs; run++) {
            double start_time = omp_get_wtime();
            matrix_vector_multiply_2d_adaptive(A, x, y, n);
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