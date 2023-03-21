#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 0.0000000000000001
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
#pragma omp parallel for default(none) shared(a, b, N, result)
    for (int i = 0; i < N; i++) {
        result[i] = a[i] - b[i];
    }
}


void mul_on_scalar(double *a, double number, int N) {
    int i;
#pragma omp parallel for default(none) shared(a, number, N)
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
#pragma omp parallel for default(none) schedule(static, matrix_size / omp_get_num_threads()) shared(result, matrix_size, vector_size, A, x)
    for (int i = 0; i < matrix_size; i++) {
        result[i] = 0;
        for (int j = 0; j < vector_size; j++) {
            result[i] += A[i * matrix_size + j] * x[j];
        }
    }
}


double euclidean_norm(double *a, int N) {
    double result = 0.0;
#pragma omp parallel for default(none) shared(a, N) reduction(+:result)
    for (int i = 0; i < N; i++) {
        result += a[i] * a[i];
    }

    return sqrt(result);
}


void vector_calculation(double *A, double *b, double *x, int N) {

    double criteria = (EPSILON) + 1.0;
    double *Ax = calloc(N, sizeof(double));

    while (criteria >= EPSILON) {
        matrix_vector_mul(A, x, Ax, N, N);

        vector_sub(Ax, b, Ax, N);
        criteria = euclidean_norm(Ax, N) / euclidean_norm(b, N);
        mul_on_scalar(Ax, TAU, N);
        vector_sub(x, Ax, x, N);
    }
    free(Ax);
}


int main(void) {

    int N = 14320;
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
    printf("Time proceeds: %lf\n", end_time - start_time);
    free(A);
    free(b);
    free(x);
    return 0;
}
