#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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
                B.data[i * B.cols + j] = -1;
            } else {
                B.data[i * B.cols + j] = 1;
            }
        }
    }
}

void print_matrix(Mat C) {
    printf("C = \n");

    for (int i = 0; i < C.rows; i++) {
        printf("( ");
        for (int j = 0; j < C.cols; j++) {
            printf("%.3lf ", C.data[i * C.cols + j]);
        }
        printf(")\n");
    }
}

void make_null(Mat C) {
    for (int i = 0; i < C.cols * C.rows; i++) {
        C.data[i] = 0.0;
    }
}

void gather_blocks(Mat A, Mat B, Mat C, Mat C_block, MPI_Comm comm_2d,
                  int *dims, int *coords, int process_num) {
    MPI_Datatype block, block_resized;
    MPI_Type_vector(A.rows / dims[0], B.cols / dims[1], B.cols,
                    MPI_DOUBLE, &block);
    MPI_Type_commit(&block);
    MPI_Type_create_resized(block, 0, (int) (B.cols / dims[1] * sizeof(double)),
                            &block_resized);
    MPI_Type_commit(&block_resized);

    int *recv_counts = calloc(process_num, sizeof(int));
    int *displs = calloc(process_num, sizeof(int));
    for (int i = 0; i < process_num; ++i) {
        recv_counts[i] = 1;
        MPI_Cart_coords(comm_2d, i, 2, coords);
        displs[i] = dims[1] * (A.rows / dims[0]) * coords[1] + coords[0];
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Gatherv(C_block.data,
                A.rows * B.cols / (dims[0] * dims[1]),
                MPI_DOUBLE,
                C.data,
                recv_counts,
                displs,
                block_resized,
                RANK_ROOT,
                comm_2d);
    MPI_Type_free(&block);
    MPI_Type_free(&block_resized);
    free(recv_counts);
    free(displs);
}


void mat_mul(Mat A, Mat B, Mat result) {
    //print_matrix(A);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 1) {
        print_matrix(B);
    }
    //Multiplication of rows of matrices corresponding to processes by a vector
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            result.data[i * B.cols + j] = 0;
            for (int k = 0; k < A.cols; k++) {
                result.data[i * B.cols + j] += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
        }
    }
    //print_matrix(result);
    MPI_Barrier(MPI_COMM_WORLD);
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

    const int columns_per_process = B.cols / grid_columns;
    const int rows_per_process = A.rows / grid_rows;


    double *data_row = calloc(A.cols * rows_per_process, sizeof(double));
    double *data_column = calloc(B.rows * columns_per_process, sizeof(double));
    double *data_result = calloc(rows_per_process * columns_per_process, sizeof(double));

    MPI_Datatype vertical_int_slice;
    MPI_Datatype vertical_int_slice_resized;
    MPI_Datatype horizontal_int_slice;

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

    if (rank_column == 0) {
        MPI_Scatter(
                /* send buffer */ B.data,
                /* number of <send data type> elements sent */ 1,
                /* send data type */ vertical_int_slice_resized,
                /* recv buffer */ data_column,
                /* number of <recv data type> elements received */ B.rows * columns_per_process,
                /* recv data type */ MPI_DOUBLE,
                                  RANK_ROOT,
                                  comm_rows
        );
    }


    MPI_Type_contiguous(A.cols * rows_per_process, MPI_DOUBLE, &horizontal_int_slice);
    MPI_Type_commit(&horizontal_int_slice);

    if (rank_row == 0) {
        MPI_Scatter(A.data, 1, horizontal_int_slice, data_row,
                    A.cols * rows_per_process, MPI_DOUBLE, RANK_ROOT, comm_columns);
    }
    MPI_Bcast(data_column, B.rows * columns_per_process, MPI_DOUBLE, RANK_ROOT,
              comm_columns);
    MPI_Bcast(data_row, A.cols * rows_per_process, MPI_DOUBLE, RANK_ROOT,
              comm_rows);

    Mat A_block = {data_row, A.cols, rows_per_process};
    Mat B_block = {data_column, columns_per_process, B.rows};
    Mat C_block = {data_result, columns_per_process, rows_per_process};
    mat_mul(A_block, B_block, C_block);
    //sleep(1 + rank);
    gather_blocks(A, B, C, C_block, comm_2d, dims, coords, size);

    MPI_Type_free(&vertical_int_slice_resized);
    MPI_Type_free(&vertical_int_slice);
    MPI_Type_free(&horizontal_int_slice);
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
    int A_size = 4, AB_size = 6, B_size = 8;
    double start_time, end_time;
    Mat A = {NULL, AB_size, A_size};
    Mat B = {NULL, B_size, AB_size};
    Mat C = {NULL, B_size, A_size};
    if (RANK_ROOT == rank) {
        A.data = (double *) malloc(A_size * AB_size * sizeof(double));
        B.data = (double *) malloc(AB_size * B_size * sizeof(double));
        C.data = (double *) calloc(A_size * B_size, sizeof(double));
        init(A, B);
    }
    for (int i = 0; i < 5; i++) {
        start_time = MPI_Wtime();
        matrix_calculation(A, B, C);
        end_time = MPI_Wtime();
        if (RANK_ROOT == rank) {
            printf("Time proceeds: %lf\n", end_time - start_time);
            print_matrix(C);
            make_null(C);
        }
    }
    if (RANK_ROOT == rank) {
        free(A.data);
        free(B.data);
        free(C.data);
    }

    MPI_Finalize();
    return 0;
}
