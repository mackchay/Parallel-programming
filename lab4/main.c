#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <math.h>
#include <string.h>

#define A 100000
#define EPSILON 0.00000001
#define D_X 2
#define D_Y 2
#define D_Z 2
#define N_X 6
#define N_Y 6
#define N_Z 6
#define MAX 1

typedef struct {
    double *values;
    int N;
    double x0;
    double y0;
    double z0;
    double D;
    int lower_edge;
    int upper_edge;
} Grid;

typedef double (*Function)(
        double x,
        double y,
        double z
);

typedef double (*GridFunction)(
        Grid grid,
        int i,
        int j,
        int k
);


Grid grid_new(
        int size,
        double x0,
        double y0,
        double z0,
        int real_size) {
    assert(size >= 1 && "Size must be >= 1");

    double *const data = (double *const) (float *) calloc(size * size * size, sizeof(*data));

    return (Grid) {
            .values = data,
            .N = size,
            .x0 = x0,
            .y0 = y0,
            .z0 = z0,
            .D = real_size,
            .lower_edge = 0,
            .upper_edge = size - 1,
    };
}


double *grid_at(
        Grid grid,
        int i,
        int j,
        int k) {
    return &grid.values[j * grid.N + i * grid.N + k];
}

Grid grid_copy(Grid grid) {
    double * const data = (double *const)calloc(grid.N * grid.N, sizeof(*data));
    memcpy(data, grid.values, sizeof(*data) * grid.N * grid.N);

    Grid copy = grid;
    copy.values = data;

    return copy;
}


void grid_free(Grid grid) {
    free(grid.values);
}

void grid_swap(
        Grid * g1,
        Grid * g2) {
    Grid tmp = *g1;
    *g1 = *g2;
    *g2 = tmp;
}

double index_x(
        Grid grid,
        int i
) {
    return grid.x0 + i * grid.D / (grid.N - 1);
}

double index_y(
        Grid grid,
        int j
) {
    return grid.y0 + j * grid.D / (grid.N - 1);
}

double index_z(
        Grid grid,
        int k
) {
    return grid.z0 + k * grid.D / (grid.N - 1);
}

double phi(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double ro(double x, double y, double z) {
    return 6 - phi(x, y, z);
}

double grid_max_diff(
        Grid grid,
        Function target) {
    double delta = 0.0f;

    for (int j = 0; j < grid.N; j += 1) {
        for (int i = 0; i < grid.N; i += 1) {
            for (int k = 0; k < grid.N; k++) {
                const double x = index_x(grid, i);
                const double y = index_y(grid, j);
                const double z = index_y(grid, k);
                const double current_delta = fabs(target(x, y, z) - *grid_at(grid, i, j, k));

                delta = fmax(delta, current_delta);
            }
        }
    }

    return delta;
}


void grid_print(Grid grid) {
    for (int j = 0; j < grid.N; j += 1) {
        for (int i = 0; i < grid.N; i += 1) {
            for (int k = 0; k < grid.N; k += 1)
                printf("%.2f ", *grid_at(grid, i, j, k));
        }
        printf("\n");
    }
}


void init_fun(Grid grid, Function fun) {
    double x, y, z;
    for (int i = 0; i < grid.N; i++) {
        x = index_x(grid, i);
        *grid_at(grid, i, grid.lower_edge, grid.lower_edge) = fun(x,
                                                                  index_y(grid, grid.lower_edge),
                                                                  index_z(grid, grid.lower_edge));
        *grid_at(grid, i, grid.upper_edge, grid.upper_edge) = fun(x,
                                                                  index_y(grid, grid.upper_edge),
                                                                  index_z(grid, grid.upper_edge));

        y = index_y(grid, i);
        *grid_at(grid, grid.lower_edge, i, grid.lower_edge) = fun(index_x(grid, grid.lower_edge),
                                                                  y,
                                                                  index_z(grid, grid.lower_edge));
        *grid_at(grid, grid.upper_edge, i, grid.upper_edge) = fun(index_x(grid, grid.upper_edge),
                                                                  y,
                                                                  index_z(grid, grid.upper_edge));
        z = index_z(grid, i);
        *grid_at(grid, grid.lower_edge, grid.lower_edge, i) = fun(index_x(grid, grid.lower_edge),
                                                                  index_y(grid, grid.lower_edge),
                                                                  z);
        *grid_at(grid, grid.upper_edge, grid.upper_edge, i) = fun(index_x(grid, grid.upper_edge),
                                                                  index_y(grid, grid.upper_edge),
                                                                  z);
    }
}

double grid_calculation(Grid grid, int i, int j, int k) {
    const double x = index_x(grid, i);
    const double y = index_y(grid, j);
    const double z = index_z(grid, k);
    const double h = grid.D / (double) (grid.N - 1);
    double hx_sq, hy_sq, hz_sq;
    hx_sq = hy_sq = hz_sq = h * h;
    const double c = 1 / (A + 2 / hx_sq + 2 / hy_sq + 2 / hz_sq);
    const double x_part = (*grid_at(grid, i - 1, j, k) + *grid_at(grid, i + 1, j, k));
    const double y_part = (*grid_at(grid, i, j - 1, k) + *grid_at(grid, i, j - 1, k));
    const double z_part = (*grid_at(grid, i, j, k - 1) + *grid_at(grid, i, j, k + 1));
    return c * (x_part + y_part + z_part - ro(x, y, z));
}

void solve(Grid grid) {
    int proc_rank, proc_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    MPI_Request req[4];
    double delta = INFINITY;
    int iter = 0;
    int layer_size = N_X / proc_num;
    int layer_x_coordinate = proc_rank * layer_size;

    Grid next = grid_copy(grid);
    while (delta > EPSILON && iter < 20) {
        printf("n_iter = %d\niter_delta = %.8f\n", iter, delta);
        grid_print(grid);
        printf("\n");
        double delta_max = 0;
        if (proc_rank != 0) {
            MPI_Isend(grid_at(grid, layer_x_coordinate, grid.upper_edge, grid.upper_edge),
                      N_Y * N_Z,
                      MPI_DOUBLE,
                      proc_rank - 1,
                      1,
                      MPI_COMM_WORLD,
                      &req[0]);
            MPI_Irecv(grid_at(grid, layer_x_coordinate, grid.upper_edge, grid.upper_edge),
                      grid.upper_edge * grid.upper_edge,
                      MPI_DOUBLE,
                      proc_rank - 1,
                      1,
                      MPI_COMM_WORLD,
                      &req[1]);
        }

        if (proc_rank != proc_num - 1) {
            MPI_Isend(grid_at(grid, layer_x_coordinate + layer_size, N_Y, N_Z),
                      grid.upper_edge * grid.upper_edge,
                      MPI_DOUBLE,
                      proc_rank + 1,
                      1,
                      MPI_COMM_WORLD,
                      &req[2]);
            MPI_Irecv(grid_at(grid, layer_x_coordinate + layer_size, grid.upper_edge, grid.upper_edge),
                      N_Y * N_Z,
                      MPI_DOUBLE,
                      proc_rank + 1,
                      1,
                      MPI_COMM_WORLD,
                      &req[3]);
        }
        for (int i = grid.lower_edge + 1; i < N_X / proc_num; i++) {
            for (int j = grid.lower_edge + 1; j < grid.upper_edge; j++) {
                for (int k = grid.lower_edge + 1; k < grid.upper_edge; k++) {
                    double old_value = *grid_at(grid, i, j, k);
                    double new_value = grid_calculation(grid, i, j, k);
                    const double tmp_delta = new_value - old_value;
                    *grid_at(next, i, j, k) = new_value;
                    delta_max = fmax(tmp_delta, delta_max);
                }
            }
        }

        if (proc_rank != proc_num - 1) {
            MPI_Wait(&req[2], MPI_STATUS_IGNORE);
            MPI_Wait(&req[3], MPI_STATUS_IGNORE);
        }
        if (proc_rank != 0) {
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        }

        grid_swap(&grid, &next);
        iter++;
        delta = delta_max;
        printf("iter = %d\n", iter);
    }
}

int main(void) {
    MPI_Init(NULL, NULL);
    int proc_num, proc_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    int layer_size = N_X / proc_num;
    int layer_x_coordinate = proc_rank * layer_size;

    if (N_X % proc_num && proc_rank == 0) {
        printf("Grid size %d should be divisible by the proc_num \n", N_X);
        return 0;
    }

    Grid grid = grid_new(N_X, -MAX + layer_x_coordinate, -MAX, -MAX, D_X);

    Function fun = phi;
    init_fun(grid, fun);
    solve(grid);

    printf("max_delta = %.8f\n", grid_max_diff(grid, fun));

    grid_free(grid);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
