#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ROOT 0

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


void vector_sub(double *a, double *b, double *result, int N) {
    for (size_t i = 0; i < N; i++) {
        result[i] = a[i] - b[i];
    }
}


void mul_on_scalar(double *a, double number, int N) {
    for (size_t i = 0; i < N; i++) {
        a[i] = number * a[i];
    }
}


void print_vector(double *x, int N) {
    printf("x = ( ");
    for (size_t i = 0; i < N; i++) {
        printf("%.3lf ", x[i]);
    }
    printf(")\n");
}


void matrix_vector_mul(double *A, double *x, double *result, int N) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (size_t i = rank * (N / size); i < rank * N / size + N / size; i++) {
        result[i] = 0;
        for (size_t j = 0; j < N; j++) {
            result[i] += A[i * N + j] * x[j];
        }
    }
    MPI_Send(result + rank * (N / size), N / size, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD);
    //printf("Process %d send information\n", rank);
    if (rank == ROOT) {
        for (int i = 0; i < size; i++) {
            MPI_Recv(result + i * (N / size), N / size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            //printf("From process %d received data.\n", i);
        }
        for (int i = N - N % size - 1; i < N; i++) {
            result[i] = 0;
            for (size_t j = 0; j < N; j++) {
                result[i] += A[i * N + j] * x[j];
            }
        }
    }

    MPI_Bcast(result, N, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
}


double scalar(double *a, int N) {
    double result = 0.0;
    for (size_t i = 0; i < N; i++) {
        result += a[i] * a[i];
    }
    return result;
}


void vector_calculation(double *A, double *b, double *x, int N) {
    double t = 0.01;
    double e = 0.00001;
    double criteria = e + 1.0;
    double *Ax = malloc(N * sizeof(double));
    double *num = malloc(N * sizeof(double));
    while (criteria >= e) {
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

    int errCode;
    if ((errCode = MPI_Init(&argc, &argv)) != 0)
    {
        return errCode;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 10;
    double *A = malloc(N * N * sizeof(double));

    double *b = malloc(N * sizeof(double));
    double *x = malloc(N * sizeof(double));

    init(A, b, x, N);
    vector_calculation(A, b, x, N);

    if (rank == ROOT) {
        print_vector(x, N);
    }


    free(A);
    free(b);
    free(x);

    MPI_Finalize();

    return 0;
}
