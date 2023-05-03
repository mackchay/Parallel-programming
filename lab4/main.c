#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <math.h>
#include <string.h>

#define RANK_ROOT 0
#define A 10000
#define EPSILON 0.000000001
#define D_X 2
#define D_Y 2
#define D_Z 2
#define N_X 8
#define N_Y 8
#define N_Z 8
#define MAX 1

typedef struct {
    double *values;
    int N;
    int N_red;
    double x0;
    double y0;
    double z0;
    double D;
    int lower_edge;
    int upper_edge;
    int upper_layer_edge;
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
        int reduced_size,
        int size,
        double x0,
        double y0,
        double z0,
        int real_size) {
    assert(size >= 1 && "Size must be >= 1");

    double *const data = (double *const) calloc(reduced_size * size * size, sizeof(double));

    return (Grid) {
            .values = data,
            .N_red = reduced_size,
            .N = size,
            .x0 = x0,
            .y0 = y0,
            .z0 = z0,
            .D = real_size,
            .lower_edge = 0,
            .upper_edge = size - 1,
            .upper_layer_edge = reduced_size - 1
    };
}


double *grid_at(
        Grid grid,
        int i,
        int j,
        int k) {
    return &grid.values[i * grid.N_red * grid.N + j * grid.N + k];
}

Grid grid_copy(Grid grid) {
    double *const data = (double *const) calloc(grid.N_red * grid.N * grid.N, sizeof(*data));
    memcpy(data, grid.values, sizeof(*data) * grid.N_red * grid.N * grid.N);

    Grid copy = grid;
    copy.values = data;

    return copy;
}


void grid_free(Grid grid) {
    free(grid.values);
}

void grid_swap(
        Grid *g1,
        Grid *g2) {
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
            for (int k = 0; k < grid.N; k += 1) {
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
    for (int i = 0; i < grid.N_red; i += 1) {
        for (int j = 0; j < grid.N; j += 1) {
            for (int k = 0; k < grid.N; k += 1) {
                printf("%lf ", *grid_at(grid, i, j, k));
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}


void init_fun(Grid grid, Function fun) {
    for (int i = grid.lower_edge; i < grid.N_red; i += 1) {
        for (int j = grid.lower_edge; j < grid.N; j += 1) {
            for (int k = grid.lower_edge; k < grid.N; k += 1) {
                if (i == grid.lower_edge || i == grid.upper_layer_edge
                    || j == grid.lower_edge || j == grid.upper_edge
                    || k == grid.lower_edge || k == grid.upper_edge) {
                    *grid_at(grid, i, j, k) = fun(index_x(grid, i),
                                                  index_y(grid, j),
                                                  index_z(grid, k));
                }
            }
        }
    }
}

double grid_calculation(Grid grid, int i, int j, int k) {
    const double x = index_x(grid, i);
    const double y = index_y(grid, j);
    const double z = index_z(grid, k);
    const double hx = grid.D / (double) (grid.N - 1);
    const double h = grid.D / (double) (grid.N - 1);
    double hx_sq, hy_sq, hz_sq;
    hx_sq = hx * hx;
    hy_sq = hz_sq = h * h;
    const double c = 1 / (A + 2 / hx_sq + 2 / hy_sq + 2 / hz_sq);
    const double x_part = (*grid_at(grid, i - 1, j, k) + *grid_at(grid, i + 1, j, k));
    const double y_part = (*grid_at(grid, i, j - 1, k) + *grid_at(grid, i, j + 1, k));
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
    int layer_x_coordinate = proc_rank * layer_size * D_X / (N_X - 1);

    Grid next = grid_copy(grid);
    while (delta > EPSILON && iter < 20) {
        if (RANK_ROOT == proc_rank) {
            printf("n_iter = %d\niter_delta = %.8f\n", iter, delta);
            //grid_print(grid);
            printf("\n");
        }

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

        printf("Staff: %lf\n", grid.x0);
        for (int i = grid.lower_edge + 1; i < grid.upper_layer_edge; i++) {
            for (int j = grid.lower_edge + 1; j < grid.upper_edge; j++) {
                for (int k = grid.lower_edge + 1; k < grid.upper_edge; k++) {
                    double old_value = *grid_at(grid, i, j, k);
                    double new_value = grid_calculation(grid, i, j, k);
                    const double tmp_delta = fabs(new_value - old_value);
                    *grid_at(next, i, j, k) = new_value;
                    delta_max = fmax(tmp_delta, delta_max);
                }
            }
        }

        printf("Staff: %lf\n", grid.x0);

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
        MPI_Allreduce(&delta_max, &delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
}

int main(void) {
    MPI_Init(NULL, NULL);
    int proc_num, proc_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    int layer_size = N_X / proc_num;
    double layer_x_coordinate = (double) proc_rank * layer_size * D_X / (N_X - 1);

    if (N_X % proc_num && proc_rank == 0) {
        printf("Grid size %d should be divisible by the proc_num \n", N_X);
        return 0;
    }

    Grid grid = grid_new(layer_size, N_X, -MAX + layer_x_coordinate, -MAX, -MAX, D_X);

    Function fun = phi;
    init_fun(grid, fun);
    solve(grid);
    Grid full_grid;
    if (RANK_ROOT == proc_rank) {
        full_grid = grid_new(N_X, N_X, -MAX, -MAX, -MAX, D_X);
        MPI_Gather(grid.values, grid.N_red * grid.N * grid.N, MPI_DOUBLE,
                   full_grid.values, full_grid.N * full_grid.N * full_grid.N, MPI_DOUBLE,
                   RANK_ROOT, MPI_COMM_WORLD);
        printf("max_delta = %.8f\n", grid_max_diff(full_grid, fun));
        grid_free(full_grid);
    }

    grid_free(grid);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
