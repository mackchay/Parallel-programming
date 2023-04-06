#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

#define RANK_ROOT 0
#define A_ROWS 3200
#define A_B_STRIP 3200
#define B_COLS 3200

typedef struct {
    double *data;
    int cols;
    int rows;
} Mat;

Mat *init_null(void) {
    Mat *new_mat = malloc(sizeof(Mat));
    new_mat->data = NULL;
    new_mat->rows = 0;
    new_mat->cols = 0;
    return new_mat;
}

Mat *init_matrix(int rows, int cols) {
    Mat *new_mat = malloc(sizeof(Mat));
    new_mat->data = calloc(rows * cols, sizeof(double));
    new_mat->rows = rows;
    new_mat->cols = cols;
    return new_mat;
}

void free_matrix(Mat *mat) {
    free(mat->data);
    free(mat);
}

void fill_matrix(Mat *A, Mat *B) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            if (i == j) {
                A->data[i * A->cols + j] = 0.0;
            } else {
                A->data[i * A->cols + j] = 1.0;
            }
        }
    }

    for (int i = 0; i < B->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            if (i == j) {
                B->data[i * B->cols + j] = -1;
            } else {
                B->data[i * B->cols + j] = 1;
            }
        }
    }
}

void print_matrix(Mat *C) {
    printf("C = \n");

    for (int i = 0; i < C->rows; i++) {
        printf("( ");
        for (int j = 0; j < C->cols; j++) {
            printf("%.3lf ", C->data[i * C->cols + j]);
        }
        printf(")\n");
    }
}

void make_null(Mat *C) {
    for (int i = 0; i < C->cols * C->rows; i++) {
        C->data[i] = 0.0;
    }
}

void gather_blocks(Mat *C, Mat *C_block, MPI_Comm comm_2d,
                   int grid_columns, int grid_rows, int process_num) {

    MPI_Datatype block, block_resized;
    MPI_Type_vector(A_ROWS / grid_rows, B_COLS / grid_columns, B_COLS,
                    MPI_DOUBLE, &block);
    MPI_Type_commit(&block);
    MPI_Type_create_resized(block, 0, (int) (B_COLS / grid_columns * sizeof(double)),
                            &block_resized);
    MPI_Type_commit(&block_resized);

    int *recv_counts = calloc(process_num, sizeof(int));
    int *displs = calloc(process_num, sizeof(int));
    int coords[2] = {0, 0};
    for (int i = 0; i < process_num; ++i) {
        recv_counts[i] = 1;
        MPI_Cart_coords(comm_2d, i, 2, coords);
        displs[i] = grid_columns * (A_ROWS / grid_rows) * coords[1] + coords[0];
    }

    MPI_Gatherv(C_block->data,
                A_ROWS * B_COLS / (grid_columns * grid_rows),
                MPI_DOUBLE,
                C->data,
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

void mat_distribution(MPI_Comm comm_2d, Mat *A, Mat *B, Mat *A_block, Mat *B_block, int columns_per_process,
                      int rows_per_process) {
    MPI_Comm comm_rows;
    MPI_Comm comm_columns;
    int grid_rank, rank_row, rank_column;
    int coords[2] = {0, 0};

    //Creating 2d entity and getting coords for every process.
    MPI_Comm_rank(comm_2d, &grid_rank);
    MPI_Cart_coords(comm_2d, grid_rank, 2, coords);
    rank_row = coords[1], rank_column = coords[0];

    //Getting groups by same row or same column.
    MPI_Comm_split(comm_2d, rank_row, rank_column, &comm_columns);
    MPI_Comm_split(comm_2d, rank_column, rank_row, &comm_rows);


    MPI_Datatype vertical_int_slice;
    MPI_Datatype vertical_int_slice_resized;
    MPI_Datatype horizontal_int_slice;

    MPI_Type_contiguous(A_B_STRIP * rows_per_process, MPI_DOUBLE, &horizontal_int_slice);
    MPI_Type_commit(&horizontal_int_slice);

    if (rank_column == 0) {
        MPI_Scatter(A->data,
                    1,
                    horizontal_int_slice,
                    A_block->data,
                    A_B_STRIP * rows_per_process,
                    MPI_DOUBLE,
                    RANK_ROOT,
                    comm_rows);
    }

    MPI_Bcast(A_block->data,
              A_B_STRIP * rows_per_process,
              MPI_DOUBLE,
              RANK_ROOT,
              comm_columns);

    MPI_Type_vector(
            /* blocks count - number of columns */ A_B_STRIP,
            /* block length  */ columns_per_process,
            /* stride - block start offset */ B_COLS,
            /* old type - element type */ MPI_DOUBLE,
            /* new type */ &vertical_int_slice
    );
    MPI_Type_commit(&vertical_int_slice);

    MPI_Type_create_resized(
            vertical_int_slice,
            /* lower bound */ 0,
            /* extent - process_num in bytes */ (int) (columns_per_process * sizeof(*B_block->data)),
            /* new type */ &vertical_int_slice_resized
    );
    MPI_Type_commit(&vertical_int_slice_resized);


    if (rank_row == 0) {
        MPI_Scatter(
                /* send buffer */ B->data,
                /* number of <send data type> elements sent */ 1,
                /* send data type */ vertical_int_slice_resized,
                /* recv buffer */ B_block->data,
                /* number of <recv data type> elements received */ A_B_STRIP * columns_per_process,
                /* recv data type */ MPI_DOUBLE,
                                  RANK_ROOT,
                                  comm_columns
        );
    }

    MPI_Bcast(B_block->data,
              A_B_STRIP * columns_per_process,
              MPI_DOUBLE,
              RANK_ROOT,
              comm_rows);

    MPI_Comm_free(&comm_columns);
    MPI_Comm_free(&comm_rows);
    MPI_Type_free(&vertical_int_slice_resized);
    MPI_Type_free(&vertical_int_slice);
    MPI_Type_free(&horizontal_int_slice);
}

void mat_mul(Mat *A, Mat *B, Mat *result) {
    //Multiplication of rows of matrices corresponding to processes by a vector
    for (int i = 0; i < result->rows; i++) {
        for (int k = 0; k < A->cols; k++) {
            for (int j = 0; j < result->cols; j++) {
                result->data[i * B->cols + j] += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
        }
    }
}

    void matrix_calculation(Mat *A, Mat *B, Mat *C) {
        int dims[2] = {0, 0}, periods[2] = {0, 0}, reorder = 0;
        int process_num, grid_columns, grid_rows;

        //communicators 2d, on rows and columns.
        MPI_Comm comm_2d;
        MPI_Comm_size(MPI_COMM_WORLD, &process_num);

        //Creating grid process_num on 2 dimensions.
        MPI_Dims_create(process_num, 2, dims);
        grid_rows = dims[1], grid_columns = dims[0];
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm_2d);

        assert(B_COLS % grid_columns == 0);
        const int columns_per_process = B_COLS / grid_columns;

        assert(A_ROWS % grid_rows == 0);
        const int rows_per_process = A_ROWS / grid_rows;

        Mat *A_block = init_matrix(rows_per_process, A_B_STRIP);
        Mat *B_block = init_matrix(A_B_STRIP, columns_per_process);
        Mat *C_block = init_matrix(rows_per_process, columns_per_process);

        mat_distribution(comm_2d, A, B, A_block, B_block, columns_per_process, rows_per_process);
        mat_mul(A_block, B_block, C_block);
        //sleep(1 + rank);
        gather_blocks(C, C_block, comm_2d, grid_columns, grid_rows, process_num);

        MPI_Comm_free(&comm_2d);
        free_matrix(A_block);
        free_matrix(B_block);
        free_matrix(C_block);
    }

    int main(int argc, char *argv[]) {

        int err_code, rank;
        err_code = MPI_Init(&argc, &argv);
        if (err_code != 0) {
            return err_code;
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        double start_time, end_time;
        Mat *A = init_null();
        Mat *B = init_null();
        Mat *C = init_null();
        if (RANK_ROOT == rank) {
            A = init_matrix(A_ROWS, A_B_STRIP);
            B = init_matrix(A_B_STRIP, B_COLS);
            C = init_matrix(A_ROWS, B_COLS);
            fill_matrix(A, B);
        }
        for (int i = 0; i < 5; i++) {
            start_time = MPI_Wtime();
            matrix_calculation(A, B, C);
            end_time = MPI_Wtime();
            if (RANK_ROOT == rank) {
                printf("Time proceeds: %lf\n", end_time - start_time);
//                print_matrix(C);
                make_null(C);
            }
        }
        if (RANK_ROOT == rank) {
            free_matrix(A);
            free_matrix(B);
            free_matrix(C);
        }

        MPI_Finalize();
        return 0;
    }
