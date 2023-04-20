#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#define a 100000
#define e 0.00000001

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

void calculateFun(void) {

}

int main(void) {

    printf("Hello, World!\n");
    return 0;
}
