#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_vector_multiply(int **A, int x[], int y[], int n) {
    for (int i = 0; i < n; i++) {
        y[i] = 0; 
        for (int j = 0; j < n; j++) {
            y[i] = y[i] + A[i][j] * x[j];
        }
    }
}

int** allocate_matrix(int n) {
    int **matrix = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        matrix[i] = malloc(n * sizeof(int));
    }
    return matrix;
}

void free_matrix(int **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
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
    
    printf("Sequential Matrix-Vector Multiplication\n");
    printf("Matrix Size\tSequential Time (ms)\n");
    printf("-------------------------------------\n");
    
    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        int **A = allocate_matrix(n);
        int *x = malloc(n * sizeof(int));
        int *y = malloc(n * sizeof(int));
        
        initialize_matrix(A, x, n);
        
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        matrix_vector_multiply(A, x, y, n);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000L + (end.tv_nsec - start.tv_nsec);
        double elapsed_ms = elapsed_ns / 1000000.0;
        
        printf("%d\t\t%.6f\n", n, elapsed_ms);
        
        free_matrix(A, n);
        free(x);
        free(y);
    }
    
    return 0;
}