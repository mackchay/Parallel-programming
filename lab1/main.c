#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void init(double *A, double *b, double *x, int N) {
    for (size_t i = 0; i < N * N; i++) {
        if (i % (N + 1) == 0) {
            A[i] = 2.0;
        } else {
            A[i] = 1.0;
        }
    }
    for (size_t i = 0; i < N; i++) {
        x[i] = 0;
        b[i] = N + 1;
    }
}


void vectorSub(double *a, double *b, double *result, int N) {
    for (size_t i = 0; i < N; i++) {
        result[i] = a[i] - b[i];
    }
}

void numberMul(double *a, double number, int N) {
    for (size_t i = 0; i < N; i++) {
        a[i] = number * a[i];
    }
}

void matrixVectorMul(double *A, double *x, double *result, int N) {
    for (size_t i = 0; i < N; i++) {
        result[i] = 0;
        for (size_t j = 0; j < N; j++) {
            result[i] += A[i * N + j] * x[j];
        }
    }
}

double scalar(double *a, int N) {
    double result = 0.0;
    for (size_t i = 0; i < N; i++) {
        result += a[i] * a[i];
    }
    return result;
}

void vectorCalculation(double *A, double *b, double *x, int N) {
    double t = 0.01;
    double e = 0.00001;
    double criteria = e + 1.0;
    double *Ax = malloc(N * sizeof(double));
    double *num = malloc(N * sizeof(double));
    while (criteria >= e) {
        matrixVectorMul(A, x, Ax, N);
        vectorSub(Ax, b, num, N);
        criteria = scalar(num, N) / scalar(b, N);
        numberMul(num, t, N);
        vectorSub(x, num, x, N);
    }
    free(num);
    free(Ax);
}

void printVector(double *x, int N) {
    printf("x = ( ");
    for (size_t i = 0; i < N; i++) {
        printf("%.3lf ", x[i]);
    }
    printf(")");
}

int main(int argc, char* argv[]) {

    int errCode;
    if ((errCode = MPI_Init(&argc, &argv)) != 0)
    {
        return errCode;
    }


    int N = 15;
    double *A = malloc(N * N * sizeof(double *));

    double *b = malloc(N * sizeof(double));
    double *x = malloc(N * sizeof(double));

    init(A, b, x, N);
    vectorCalculation(A, b, x, N);
    printVector(x, N);


    free(A);
    free(b);
    free(x);

    MPI_Finalize();

    return 0;
}
