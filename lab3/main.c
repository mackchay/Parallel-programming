#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define RANK_ROOT 0

typedef struct {
    double *data;
    int cols;
    int rows;
} Mat;

void init(Mat A, Mat B) {
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (i == j) {
                A.data[i * A.cols + j] = 0.0;
            } else {
                A.data[i * A.cols + j] = 1.0;
            }
        }
    }

    for (int i = 0; i < B.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            if (i == j) {
                B.data[i * B.cols + j] = -0.5;
            } else {
                B.data[i * B.cols + j] = 0.5;
            }
        }
    }
}

void make_null(double *C, int A_size, int B_size) {
    for (int i = 0; i < A_size * B_size; i++) {
        C[i] = 0.0;
    }
}

void generate_counts_and_displs(int *counts, int *displs, int size, int processes) {
    int chunk_size = processes / size;       // Elements in each process.
    int remainder = processes % size;        // Remains from division

    for (int i = 0; i < size; ++i) {
        counts[i] = chunk_size;
        if (i < remainder) {
            counts[i]++;
        }
    }

    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i-1] + counts[i-1];
    }
}

void print_matrix(double *C, int A_size, int B_size) {
    printf("C = \n");

    for (int i = 0; i < A_size; i++) {
        printf("( ");
        for (int j = 0; j < B_size; j++) {
            printf("%.3lf ", C[i * A_size + j]);
        }
        printf(")\n");
    }
}


void mat_mul(Mat A, Mat B, Mat result) {
    //Multiplication of rows of matrices corresponding to processes by a vector
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            result.data[i * result.cols + j] = 0;
            for (int k = 0; k < A.rows; k++) {
                result.data[i * result.cols + j] += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
        }
    }
}

void matrix_calculation(Mat A, Mat B, Mat C) {
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2], reorder = 1;
    int size, rank, grid_columns, grid_rows;
    int rank_row, rank_column;

    //communicators 2d, on rows and columns.
    MPI_Comm comm_2d;
    MPI_Comm comm_rows;
    MPI_Comm comm_columns;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Creating grid size on 2 dimensions.
    MPI_Dims_create(size, 2, dims);
    grid_rows = dims[0];
    grid_columns = dims[1];

    //Creating 2d entity and getting coords for every process.
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm_2d);
    MPI_Cart_get(comm_2d, 2, dims, periods, coords);
    rank_row = coords[0], rank_column = coords[1];

    //Getting groups by same row or same column.
    MPI_Comm_split(comm_2d, rank_row, rank_column, &comm_columns);
    MPI_Comm_split(comm_2d, rank_column, rank_row, &comm_rows);

    int *counts_row = malloc(grid_rows * sizeof(int));
    int *displs_row = malloc(grid_rows * sizeof(int));
    int *counts_columns = malloc(grid_columns * sizeof(int));
    int *displs_columns = malloc(grid_columns * sizeof(int));

    if (RANK_ROOT == rank) {
        generate_counts_and_displs(counts_row, displs_row, A.rows, grid_rows);
        generate_counts_and_displs(counts_columns, displs_columns, B.cols,
                                   grid_columns);
    }

    const int columns_per_process = B.cols / grid_columns;
    const int rows_per_process = A.rows / grid_rows;


    double *data_row = calloc(A.cols * rows_per_process, sizeof(double));
    double *data_column = calloc(B.rows * columns_per_process, sizeof(double));
    double *data_result = calloc(rows_per_process * columns_per_process, sizeof(double));

    MPI_Datatype vertical_int_slice;
    MPI_Datatype vertical_int_slice_resized;
    MPI_Datatype horizontal_int_slice;
    MPI_Datatype horizontal_int_slice_resized;

    MPI_Type_vector(
            /* blocks count - number of columns */ B.rows,
            /* block length  */ columns_per_process,
            /* stride - block start offset */ B.cols,
            /* old type - element type */ MPI_DOUBLE,
            /* new type */ &vertical_int_slice
    );
    MPI_Type_commit(&vertical_int_slice);

    MPI_Type_create_resized(
            vertical_int_slice,
            /* lower bound */ 0,
            /* extent - size in bytes */ (int) (columns_per_process * sizeof(*data_column)),
            /* new type */ &vertical_int_slice_resized
    );
    MPI_Type_commit(&vertical_int_slice_resized);

    if (0 == rank_row) {
        MPI_Scatter(
                /* send buffer */ B.data ,
                /* number of <send data type> elements sent */ 1,
                /* send data type */ vertical_int_slice_resized,
                /* recv buffer */ data_column,
                /* number of <recv data type> elements received */ B.rows * columns_per_process,
                /* recv data type */ MPI_DOUBLE,
                                  RANK_ROOT,
                                  comm_rows
        );
    }
    MPI_Type_contiguous(A.cols, MPI_DOUBLE, &horizontal_int_slice);
    MPI_Type_commit(&horizontal_int_slice);
    MPI_Type_create_resized(horizontal_int_slice, 0, (int)(rows_per_process*sizeof(*data_row)),
                            &horizontal_int_slice_resized);
    MPI_Type_commit(&horizontal_int_slice_resized);
    if (0 == rank_column) {
        MPI_Scatter(A.data, 1, horizontal_int_slice_resized,data_row,
                    A.cols * rows_per_process, MPI_DOUBLE, RANK_ROOT, comm_columns);

    }
    MPI_Bcast(data_column, B.rows * columns_per_process, MPI_DOUBLE, RANK_ROOT,
              comm_rows);
    MPI_Bcast(data_row, A.cols * rows_per_process, MPI_DOUBLE, RANK_ROOT,
              comm_columns);

    Mat ABlock = {data_row, A.cols, rows_per_process};
    Mat BBlock = {data_column, columns_per_process, B.rows};
    Mat CBlock = {data_result, columns_per_process, rows_per_process};
    mat_mul(ABlock, BBlock, CBlock);
    //sleep(1 + rank);
    printf("RANK %d:\n", rank);
    //mat_print((Mat) {data, .columns = N, .cols = columns_per_process});
    MPI_Gather(data_result, columns_per_process, MPI_DOUBLE, C.data, C.cols * C.rows,
               MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

    MPI_Type_free(&vertical_int_slice_resized);
    MPI_Type_free(&vertical_int_slice);
    MPI_Type_free(&horizontal_int_slice);
    MPI_Type_free(&horizontal_int_slice_resized);
    free(data_column);
    free(data_row);
    free(data_result);
}

int main(int argc, char *argv[]) {

    int err_code, rank;
    err_code = MPI_Init(&argc, &argv);
    if (err_code != 0) {
        return err_code;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int A_size = 10, AB_size = 14, B_size = 12;
    double start_time, end_time;
    Mat A = {NULL, A_size, AB_size};
    Mat B = {NULL, AB_size, B_size};
    Mat C = {NULL, A_size, B_size};
    if (RANK_ROOT == rank) {
        A.data = (double *) malloc(A_size * AB_size * sizeof(double));
        B.data = (double *) malloc(AB_size * B_size * sizeof(double));
        C.data = (double *) malloc(A_size * B_size * sizeof(double));
        init(A, B);
    }
    for (int i = 0; i < 5; i++) {
        start_time = MPI_Wtime();
        matrix_calculation(A, B, C);
        end_time = MPI_Wtime();
        printf("Time proceeds: %lf\n", end_time - start_time);
        print_matrix(C.data, A_size, B_size);
        make_null(C.data, A_size, B_size);
    }
    if (RANK_ROOT == rank) {
        free(A.data);
        free(B.data);
        free(C.data);
    }

    MPI_Finalize();
    return 0;
}
