#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 0.0000000000000000001
#define TAU 0.0001

void init(double *A, double *b, int N) {

    for (int i = 0; i < N * N; i++) {
        if (i % (N + 1) == 0) {
            A[i] = 2.0;
        } else {
            A[i] = 1.0;
        }
    }

    for (int i = 0; i < N; i++) {
        b[i] = N + 1;
    }
}


void vector_sub(double *a, double *b, double *result, int N) {
    int i;
    for (i = 0; i < N; i++) {
        result[i] = a[i] - b[i];
    }
}


void mul_on_scalar(double *a, double number, int N) {
    int i;
    for (i = 0; i < N; i++) {
        a[i] = number * a[i];
    }
}


void print_vector(double *x, int N) {
    printf("x = ( ");

    for (int i = 0; i < N; i++) {
        printf("%.3lf ", x[i]);
    }
    printf(")\n");
}


void matrix_vector_mul(double *A, double *x, double *result, int matrix_size, int vector_size) {
    //Multiplication of rows of matrices corresponding to processes by a vector
    int i, j;
    for (i = 0; i < matrix_size; i++) {
        for (j = 0; j < vector_size; j++) {
            result[i] += A[i * matrix_size + j] * x[j];
        }
    }
}


double euclidean_norm(double *a, int N) {
    double result;
    result = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        result += a[i] * a[i];
    }
    result = sqrt(result);
    return result;
}


void vector_calculation(double *A, double *b, double *x, int N) {

    double criteria = (EPSILON) + 1.0;
    double *Ax = calloc(N, sizeof(double));
    double norm = euclidean_norm(Ax, N);
    double norm_b = euclidean_norm(b, N);
    int i, j;
#pragma omp parallel private(i, j)
    {
        while (criteria >= EPSILON) {
#pragma omp parallel for shared(Ax, A) private(j)
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    Ax[i] += A[i * N + j] * x[j];
                }
            }
#pragma omp parallel for shared(Ax) private(j)
            for (i = 0; i < N; i++) {
                Ax[i] = Ax[i] - b[i];
            }
#pragma omp atomic write
            norm = 0;

#pragma omp parallel for shared(Ax) reduction (+:norm)
            for (i = 0; i < N; i++) {
                norm += Ax[i] * Ax[i];
            }

#pragma omp atomic write
            norm = sqrt(norm);

#pragma omp parallel for shared(Ax)
            for (i = 0; i < N; i++) {
                Ax[i] = TAU * Ax[i];
            }
#pragma omp parallel for
            for (i = 0; i < N; i++) {
                x[i] = Ax[i] - x[i];
            }
#pragma omp critical
            criteria = norm / norm_b;
        }
    }
    free(Ax);
}


int main(void) {

    int N = 200;
    double start_time, end_time;

    double *A = NULL;
    double *b = calloc(N, sizeof(double));
    double *x = calloc(N, sizeof(double));


    A = calloc(N * N, sizeof(double));
    init(A, b, N);

    start_time = omp_get_wtime();
    vector_calculation(A, b, x, N);
    end_time = omp_get_wtime();

    print_vector(x, N);
    printf("Time proceeds: %lf", end_time - start_time);
    free(A);
    free(b);
    free(x);
    return 0;
}

