#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <assert.h>


#define RANK_ROOT 0


typedef struct {
    int *data;
    int cols;
    int rows;
} Mat;


void mat_fill(
        Mat mat,
        int comm_size) {
    assert(mat.cols % comm_size == 0);

    const int columns_per_process = mat.cols / comm_size;

    for (int y = 0; y < mat.rows; y += 1) {
        for (int proc_index = 0; proc_index < comm_size; proc_index += 1) {
            for (int x = columns_per_process * proc_index; x < columns_per_process * (proc_index + 1); x += 1) {
                mat.data[y * mat.cols + x] = proc_index;
            }
        }
    }
}

void mat_mul(double *A, double *B, double *result, int A_size, int AB_size, int B_size) {
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

void mat_print(Mat mat) {
    for (int i = 0; i < mat.rows * mat.cols; i += 1) {
        printf("%d ", mat.data[i]);

        if ((i + 1) % mat.cols == 0) {
            printf("\n");
        }
    }
}


int run(
        int size,
        int rank) {
    const int N = 8;
    assert(N % size == 0);


    const int columns_per_process = N / size;

    int *data0 = NULL;
    int *data = NULL;

    if (RANK_ROOT == rank) {
        data0 = (int *) calloc(N * N, sizeof(*data));
        mat_fill((Mat) {data0, N, N}, size);
        printf("ROOT:\n");
        mat_print((Mat) {data0, N, N});
    }

    data = (int *) calloc(N * columns_per_process, sizeof(*data));


    MPI_Datatype vertical_int_slice;
    MPI_Datatype vertical_int_slice_resized;
    MPI_Type_vector(
            /* blocks count - number of rows */ N,
            /* block length  */ columns_per_process,
            /* stride - block start offset */ N,
            /* old type - element type */ MPI_INT,
            /* new type */ &vertical_int_slice
    );
    MPI_Type_commit(&vertical_int_slice);

    MPI_Type_create_resized(
            vertical_int_slice,
            /* lower bound */ 0,
            /* extent - size in bytes */ (int) (columns_per_process * sizeof(*data)),
            /* new type */ &vertical_int_slice_resized
    );
    MPI_Type_commit(&vertical_int_slice_resized);

    MPI_Scatter(
            /* send buffer */ data0,
            /* number of <send data type> elements sent */ 1,
            /* send data type */ vertical_int_slice_resized,
            /* recv buffer */ data,
            /* number of <recv data type> elements received */ N * columns_per_process,
            /* recv data type */ MPI_INT,
                              RANK_ROOT,
                              MPI_COMM_WORLD
    );

    sleep(1 + rank);
    printf("RANK %d:\n", rank);
    mat_print((Mat) {data, .rows = N, .cols = columns_per_process});

    MPI_Type_free(&vertical_int_slice_resized);
    MPI_Type_free(&vertical_int_slice);
    free(data);
    free(data0);
    return EXIT_SUCCESS;
}


int main(
        int argc,
        char **argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int exit_code = run(size, rank);

    MPI_Finalize();

    return exit_code;
}


