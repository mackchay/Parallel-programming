#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define RANK_ROOT 0

typedef struct {
    int *data;
    int cols;
    int rows;
} Mat;

void init(double *A, double *B, double *C, int A_size, int AB_size, int B_size) {
    for (int i = 0; i < A_size; i++) {
        for (int j = 0; j < AB_size; j++) {
            if (i == j) {
                A[i * AB_size + j] = 0.0;
            } else {
                A[i * AB_size + j] = 1.0;
            }
        }
    }

    for (int i = 0; i < AB_size; i++) {
        for (int j = 0; j < B_size; j++) {
            if (i == j) {
                B[i * B_size + j] = -0.5;
            } else {
                B[i * B_size + j] = 0.5;
            }
        }
    }
    for (int i = 0; i < A_size * B_size; i++) {
        C[i] = 0.0;
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


void mat_mul(double *A, double *B, double *result, int A_size, int AB_size, int B_size) {
    //Multiplication of rows of matrices corresponding to processes by a vector
    for (int i = 0; i < A_size; i++) {
        for (int j = 0; j < B_size; j++) {
            result[i * B_size + j] = 0;
            for (int k = 0; k < AB_size; k++) {
                result[i * B_size + j] += A[i * AB_size + k] * B[k * B_size + j];
            }
        }
    }
}

void mat_fill(
        Mat mat,
        int comm_size) {

    const int columns_per_process = mat.cols / comm_size;

    for (int y = 0; y < mat.rows; y += 1) {
        for (int proc_index = 0; proc_index < comm_size; proc_index += 1) {
            for (int x = columns_per_process * proc_index; x < columns_per_process * (proc_index + 1); x += 1) {
                mat.data[y * mat.cols + x] = proc_index;
            }
        }
    }
}

void matrix_calculation(double *A, double *B, double *C, int rows, int strip_size, int columns) {
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2], reorder = 1;
    int size, rank, grid_columns, grid_rows;
    int rank_row, rank_column;
    C[0] = 0;

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
        generate_counts_and_displs(counts_row, displs_row, rows, grid_rows);
        generate_counts_and_displs(counts_columns, displs_columns, columns,
                                   grid_columns);
    }

    const int columns_per_process = columns / grid_columns;
    const int rows_per_process = rows / grid_rows;


    double *data_row = calloc(strip_size * rows_per_process, sizeof(double));
    double *data_column = calloc(strip_size * columns_per_process, sizeof(double));

    MPI_Datatype vertical_int_slice;
    MPI_Datatype vertical_int_slice_resized;
    MPI_Datatype horizontal_int_slice;
    MPI_Datatype horizontal_int_slice_resized;

    MPI_Type_vector(
            /* blocks count - number of columns */ strip_size,
            /* block length  */ columns_per_process,
            /* stride - block start offset */ columns,
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
                /* send buffer */ B ,
                /* number of <send data type> elements sent */ 1,
                /* send data type */ vertical_int_slice_resized,
                /* recv buffer */ data_column,
                /* number of <recv data type> elements received */ strip_size * columns_per_process,
                /* recv data type */ MPI_DOUBLE,
                                  RANK_ROOT,
                                  comm_rows
        );
    }
    MPI_Type_contiguous(strip_size, MPI_DOUBLE, &horizontal_int_slice);
    MPI_Type_commit(&horizontal_int_slice);
    MPI_Type_create_resized(horizontal_int_slice, 0, (int)(rows_per_process*sizeof(*data_row)),
                            &horizontal_int_slice_resized);
    MPI_Type_commit(&horizontal_int_slice_resized);
    if (0 == rank_column) {
        MPI_Scatter(A, 1, horizontal_int_slice_resized,data_row,
                    strip_size * rows_per_process, MPI_DOUBLE, RANK_ROOT, comm_columns);
    }
    MPI_Bcast(data_column, strip_size * columns_per_process, MPI_DOUBLE, RANK_ROOT,
              comm_rows);
    MPI_Bcast(data_row, strip_size * rows_per_process, MPI_DOUBLE, RANK_ROOT,
              comm_columns);
    mat_mul(data_row, data_column, C, rows_per_process,
            strip_size, columns_per_process);
    //sleep(1 + rank);
    printf("RANK %d:\n", rank);
    //mat_print((Mat) {data, .columns = N, .cols = columns_per_process});

    MPI_Type_free(&vertical_int_slice_resized);
    MPI_Type_free(&vertical_int_slice);
    MPI_Type_free(&horizontal_int_slice);
    MPI_Type_free(&horizontal_int_slice_resized);
    free(data_column);
    free(data_row);
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
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    if (RANK_ROOT == rank) {
        A = (double *) malloc(A_size * AB_size * sizeof(double));
        B = (double *) malloc(AB_size * B_size * sizeof(double));
        C = (double *) malloc(A_size * B_size * sizeof(double));
        init(A, B, C, A_size, AB_size, B_size);
    }
    for (int i = 0; i < 5; i++) {
        start_time = MPI_Wtime();
        matrix_calculation(A, B, C, A_size, AB_size, B_size);
        end_time = MPI_Wtime();
        printf("Time proceeds: %lf\n", end_time - start_time);
        print_matrix(C, A_size, B_size);
        make_null(C, A_size, B_size);
    }
    if (RANK_ROOT == rank) {
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();
    return 0;
}
