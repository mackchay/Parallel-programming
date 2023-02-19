#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define RANK_ROOT 0
#define TAG 0
#define EPSILON 0.00001
#define TAU 0.01

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
    for (int i = 0; i < N; i++) {
        result[i] = a[i] - b[i];
    }
}


void mul_on_scalar(double *a, double number, int N) {
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


void matrix_vector_mul(double *A, double *x, double *result, int N) {
    //Multiplication of rows of matrices corresponding to processes by a vector
        for (int i = 0; i < N; i++) {
            result[i] += A[i] * x[i];
        }
}


double euclidean_norm(double *a, int N) {
    double result = 0.0;
    for (int i = 0; i < N; i++) {
        result += a[i] * a[i];
    }

    return sqrt(result);
}


void vector_calculation(double *A, double *b, double *x, int N) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double criteria = (EPSILON) + 1.0;
    double *Ax = calloc(N, sizeof(double));
    double *AxBuf = calloc(N / size, sizeof(double));
    double *ABuf = calloc(N * (N / size), sizeof(double));

    MPI_Bcast(b, N, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    MPI_Bcast(x, N, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    MPI_Scatter(A, N * (N / size), MPI_DOUBLE, ABuf, N * (N / size),
                    MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
    while (criteria >= EPSILON) {
        MPI_Scatter(Ax, N / size, MPI_DOUBLE, AxBuf, N / size,
                    MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
        matrix_vector_mul(ABuf, x, AxBuf, N * (N / size));
        vector_sub(AxBuf, b, AxBuf, N / size);
        MPI_Allgather(AxBuf, N / size, MPI_DOUBLE, Ax, N / size,
                      MPI_DOUBLE, MPI_COMM_WORLD);

        if (rank == RANK_ROOT) {
            criteria = euclidean_norm(Ax, N) / euclidean_norm(b, N);
            mul_on_scalar(Ax, TAU, N);
            vector_sub(x, Ax, x, N);
        }
        MPI_Bcast(x, N, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    free(Ax);
    free(AxBuf);
    free(ABuf);
}


int main(int argc, char* argv[]) {

    int errCode, rank, N = 100;
    if ((errCode = MPI_Init(&argc, &argv)) != 0)
    {
        return errCode;
    }
    double begin, end;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double *A = NULL;
    double *b = calloc(N, sizeof(double));
    double *x = calloc(N, sizeof(double));

    if (rank == RANK_ROOT) {
        A = calloc(N * N, sizeof (double));
        init(A, b, N);
    }

    begin = MPI_Wtime();
    vector_calculation(A, b, x, N);
    end = MPI_Wtime();

    if (rank == RANK_ROOT) {
        print_vector(x, N);
        printf("The elapsed time is %lf seconds", end - begin);
    }

    free(A);
    free(b);
    free(x);
    MPI_Finalize();
    return 0;
}
