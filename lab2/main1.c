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
#pragma omp for
    for (int i = 0; i < N; i++) {
        result[i] = a[i] - b[i];
    }
}


void mul_on_scalar(double *a, double number, int N) {
#pragma omp for
    for (int i = 0; i < N; i++) {
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
#pragma omp for schedule(static, matrix_size / omp_get_num_threads())
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < vector_size; j++) {
            result[i] += A[i * matrix_size + j] * x[j];
        }
    }
}


double euclidean_norm(double *a, int N) {
    double result;
    result = 0.0;

#pragma omp parallel for default(none) shared(N, a) reduction(+:result)
    for (int i = 0; i < N; i++) {
        result += a[i] * a[i];
    }
    result = sqrt(result);
    return result;
}


void vector_calculation(double *A, double *b, double *x, int N) {

    double criteria = (EPSILON) + 1.0;
    double norm_Ax = 0;
    double norm_b = euclidean_norm(b, N);
    double *Ax;
#pragma omp parallel default(none) private(Ax) shared(A, x, b, criteria, norm_Ax, norm_b, N)
    {
        Ax = calloc(N, sizeof(double));
        while (criteria >= EPSILON) {
            matrix_vector_mul(A, x, Ax, N, N);
            vector_sub(Ax, b, Ax, N);

#pragma omp single
            {
                norm_Ax = euclidean_norm(Ax, N);
                criteria = norm_Ax / norm_b;
            }
            mul_on_scalar(Ax, TAU, N);
            vector_sub(x, Ax, x, N);
        }
        free(Ax);
    }
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

