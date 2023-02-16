#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define RANK_ROOT 0
#define TAG 0
#define EPSILON 0.000000001

void init(double *A, double *b, double *x, int N) {
    for (int i = 0; i < N * N; i++) {
        if (i % (N + 1) == 0) {
            A[i] = 2.0;
        } else {
            A[i] = 1.0;
        }
    }
    for (int i = 0; i < N; i++) {
        x[i] = 0;
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
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Multiplication of rows of matrices corresponding to processes by a vector
    for (int i = rank * (N / size); i < rank * N / size + N / size; i++) {
        result[i] = 0;
        for (int j = 0; j < N; j++) {
            result[i] += A[i * N + j] * x[j];
        }
    }
    if (rank != RANK_ROOT) {
        MPI_Send(result + rank * (N / size), N / size, MPI_DOUBLE,
                 RANK_ROOT, TAG, MPI_COMM_WORLD);
    }

    //Root process accepts values from other processors
    if (rank == RANK_ROOT) {
        for (int other_rank = RANK_ROOT + 1; other_rank < size; other_rank++) {
            MPI_Recv(result + other_rank * (N / size), N / size, MPI_DOUBLE, other_rank, TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
        //Work with remaining unallocated part by processes.
        for (int i = N - N % size; i < N; i++) {
            result[i] = 0;
            for (int j = 0; j < N; j++) {
                result[i] += A[i * N + j] * x[j];
            }
        }
    }
    MPI_Bcast(result, N, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
}


double scalar(double *a, int N) {
    double result = 0.0;
    for (int i = 0; i < N; i++) {
        result += a[i] * a[i];
    }

    return sqrt(result);
}


void vector_calculation(double *A, double *b, double *x, int N) {
    double t = 0.01;
    double criteria = (EPSILON) + 1.0;
    double *Ax = calloc(N, sizeof(double));
    double *num = calloc(N, sizeof(double));
    while (criteria >= EPSILON) {
        matrix_vector_mul(A, x, Ax, N);
        vector_sub(Ax, b, num, N);
        criteria = scalar(num, N) / scalar(b, N);
        mul_on_scalar(num, t, N);
        vector_sub(x, num, x, N);
    }
    free(num);
    free(Ax);
}


int main(int argc, char* argv[]) {

    int errCode, rank, N = 200;
    if ((errCode = MPI_Init(&argc, &argv)) != 0)
    {
        return errCode;
    }
    clock_t begin, end;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double *A = calloc(N * N, sizeof(double));
    double *b = calloc(N, sizeof(double));
    double *x = calloc(N, sizeof(double));

    begin = clock();
    init(A, b, x, N);
    vector_calculation(A, b, x, N);
    if (rank == RANK_ROOT) {
        print_vector(x, N);
        end = clock();
        printf("The elapsed time is %lf seconds", (double)(end - begin) / CLOCKS_PER_SEC);
    }

    free(A);
    free(b);
    free(x);
    MPI_Finalize();
    return 0;
}
