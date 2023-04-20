#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <math.h>

#define A 100000
#define EPSILON 0.00000001

#define MIN -1
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

    double * const data = (double *const) (float *) calloc(size * size * size, sizeof(*data));

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


double * grid_at(
        Grid grid,
        int i,
        int j,
        int k) {
    return &grid.values[j * grid.N + i * grid.N + k];
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
    return x*x + y*y + z*z;
}

double ro(double x, double y, double z) {
    return 6 - phi(x, y, z);
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
    return c * (x_part + y_part + z_part);
}

void solve(Grid grid) {
    double delta = INFINITY;
    int iter = 0;
    while (delta < EPSILON && iter < 20) {
        for (int i = grid.lower_edge + 1; i < grid.upper_edge; i++) {
            for (int j = grid.lower_edge + 1; j < grid.upper_edge; j++) {
                for (int k = grid.lower_edge + 1; k < grid.upper_edge; k++) {
                    double old_value = *grid_at(grid, i, j, k);
                    double new_value = grid_calculation(grid, i, j, k);
                }
            }
        }
    }
}

int main(void) {

    printf("Hello, World!\n");
    return 0;
}
