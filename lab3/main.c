#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 0.0000000000000001
#define TAU 0.0001

void init(double *A, double *B, double *C, int A_size, int AB_size, int B_size) {
    for (int i = 0; i < A_size; i++) {
        for (int j = 0; j < AB_size; j++) {
            if (i == j) {
                A[i] = 0.0;
            } else {
                A[i] = 1.0;
            }
        }
    }
    for (int i = 0; i < AB_size * A_size; i++) {
        for (int j = 0; j < B_size; j++) {
            if (i == j) {
                B[i] = -0.5;
            } else {
                B[i] = 0.5;
            }
        }
    }
}


void vector_sub(double *a, double *b, double *result, int N) {
    for (int i = 0; i < N; i++) {
        result[i] = a[i] - b[i];
    }
}


void mul_on_scalar(double *a, double number, int N) {
    int i;
    for (i = 0; i < N; i++) {
        a[i] = number * a[i];
    }
}


void print_matrix(double *x, int A_size, int B_size) {
    printf("C = \n");

    for (int i = 0; i < A_size; i++) {
        printf("( ");
        for (int j; j < B_size; j++) {
            printf("%.3lf ", x[i * A_size + j]);
        }
        printf(")\n");
    }
    printf(")\n");
}


void matrix_vector_mul(double *A, double *B, double *result, int A_size, int AB_size, int B_size) {
    //Multiplication of rows of matrices corresponding to processes by a vector
    for (int i = 0; i < A_size; i++) {
        for (int j = 0; j < B_size; j++) {
            result[i * A_size + j] = 0;
            for (int k = 0; k < AB_size; k++) {
                result[i * A_size + j] += A[i * A_size + k] * B[k * AB_size + j];
            }
        }
    }
}

//double euclidean_norm(double *a, int N) {
//    double result = 0.0;
//    for (int i = 0; i < N; i++) {
//        result += a[i] * a[i];
//    }
//
//    return sqrt(result);
//}


void matrix_calculation(double *A, double *B, double *C, int A_size, int AB_size, int B_size) {

    matrix_vector_mul(A, B, C, A_size, AB_size, B_size);

}


int main(void) {

    int N = 14320;
    int A_size = 10, AB_size = 15, B_size = 20;
    double start_time, end_time;

    double *A = calloc(A_size * AB_size, sizeof(double));
    double *B = calloc(AB_size * B_size, sizeof(double));
    double *C = calloc(A_size * B_size, sizeof(double));;
    init(A, B, C, A_size, AB_size, B_size);
//    start_time = omp_get_wtime();
    matrix_calculation(A, B, C, A_size, AB_size, B_size);
//    end_time = omp_get_wtime();

    print_matrix(C, A_size, B_size);
    printf("Time proceeds: %lf\n", end_time - start_time);
    free(A);
    free(B);
    free(C);
    return 0;
}

