#include<mpi.h>
#include<math.h>
#include<stdio.h>
#include<float.h>
#include<malloc.h>
#include<string.h>

#define N_X 640
#define N_Y 640
#define N_Z 640
#define A 10e5
#define EPSILON 10e-8
#define D_X 2
#define D_Y 2
#define D_Z 2
#define X_0 -1
#define Y_0 -1
#define Z_0 -1
#define RANK_ROOT 0

#define index(i, j, k) N_X * N_Y * i + N_Y * j + k

double fi(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double ro(double x, double y, double z) {
    return 6 - A * fi(x, y, z);
}

void initializeLayer(int sizeLayer, double* curLayer, int rank) {
    for (int i = 0; i < sizeLayer + 2; i++) {
        int coordZ = i + ((rank * sizeLayer) - 1);
        double z = Z_0 + coordZ * D_Z / (double)(N_Z - 1);
        for (int j = 0; j < N_X; j++) {
            double x = (X_0 + j * D_X / (double)(N_X - 1));
            for (int k = 0; k < N_Y; k++) {
                double y = Y_0 + k * D_Y / (double)(N_Y - 1);
                if (k != 0 && k != N_Y - 1 && j != 0 && j != N_X - 1 && z != Z_0 && z != Z_0 + D_Z) {
                    curLayer[index(i, j, k)] = 0;
                }
                else {
                    curLayer[index(i, j, k)] = fi(x, y, z);
                }
            }
        }
    }
}

void printData(double* data) {
    for (int i = 0; i < N_Z; i++) {
        for (int j = 0; j < N_X; j++) {
            for (int k = 0; k < N_Y; k++) {
                printf(" %7.4f", data[index(i, j, k)]);
            }
            printf(";");
        }
        printf("\n");
    }
}

double calculateDelta(double* area) {
    double deltaMax = DBL_MIN;
    double x, y, z;
    for(int i = 0; i < N_X; i++) {
        x = X_0 + i * D_X / (double)(N_X - 1);
        for (int j = 0; j < N_Y; j++) {
            y = Y_0 + j * D_Y / (double)(N_Y - 1);
            for (int k = 0; k < N_Z; k++) {
                z = Z_0 + k * D_Z / (double)(N_Z - 1);
                deltaMax = fmax(deltaMax, fabs(area[index(k, i, j)] - fi(x, y, z)));
            }
        }
    }
    return deltaMax;
}

double calculateLayer(int coordZ, int layerNumber, double* prevLayer, double* curLayer) {
    int curCoordZ = coordZ + layerNumber;
    double deltaMax = DBL_MIN;
    double x, y, z;

    if (curCoordZ == 0 || curCoordZ == N_Z - 1) {
        memcpy(curLayer + layerNumber * N_X * N_Y, prevLayer + layerNumber * N_X * N_Y, N_X * N_Y * sizeof(double));
        deltaMax = 0;
    }
    else {
        z = Z_0 + curCoordZ * D_Z / (double)(N_Z - 1);
        for (int i = 0; i < N_X; i++) {
            x = X_0 + i * D_X / (double)(N_X - 1);
            for (int j = 0; j < N_Y; j++) {
                y = Y_0 + j * D_Y / (double)(N_Y - 1);
                if (i == 0 || i == N_X - 1 || j == 0 || j == N_Y - 1) {
                    curLayer[index(layerNumber, i, j)] = prevLayer[index(layerNumber, i, j)];
                }
                else {
                    double H_X = D_X / (double)(N_X - 1);
                    double H_Y = D_Y / (double)(N_Y - 1);
                    double H_Z = D_Z / (double)(N_Z - 1);
                    curLayer[index(layerNumber, i, j)] =
                            ((prevLayer[index(layerNumber + 1, i, j)] + prevLayer[index(layerNumber - 1, i, j)]) / (H_Z * H_Z) +
                             (prevLayer[index(layerNumber, i + 1, j)] + prevLayer[index(layerNumber, i - 1, j)]) / (H_X * H_X) +
                             (prevLayer[index(layerNumber, i, j + 1)] + prevLayer[index(layerNumber, i, j - 1)]) / (H_Y * H_Y) -
                             ro(x, y, z)) / (2 / (H_X * H_X) + 2 / (H_Y * H_Y) + 2 / (H_Z * H_Z) + A);

                    if (fabs(curLayer[index(layerNumber, i, j)] - prevLayer[index(layerNumber, i, j)]) > deltaMax)
                        deltaMax = curLayer[index(layerNumber, i, j)] - prevLayer[index(layerNumber, i, j)];
                }
            }
        }
    }
    return deltaMax;
}

int main(int argc, char* argv[]) {
    int size = 0;
    int rank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request req[4];

    if ((N_X % size || N_Y % size || N_Z % size) && rank == 0) {
        printf("Error size\n");
        return 0;
    }

    double* area = NULL;
    double globalMaxDelta = DBL_MAX;
    int layerSize = N_Z / size;
    int layerZCoord = rank * layerSize - 1;

    int usedLayerSize = (layerSize + 2) * N_X * N_Y;
    double* prevLayer = (double*)malloc(usedLayerSize * sizeof(double));
    double* curLayer = (double*)malloc(usedLayerSize * sizeof(double));
    initializeLayer(layerSize, prevLayer, rank);

    int iter = 0;
    double start = MPI_Wtime();
    while (globalMaxDelta > EPSILON) {
        double procMaxDelta = DBL_MIN;
        double tmpMaxDelta;
        iter++;
        printf("iter: %d\n", iter);


        if (rank != 0) {
            MPI_Isend(curLayer + N_X * N_Y, N_X * N_Y, MPI_DOUBLE,
                      rank - 1, 1, MPI_COMM_WORLD, &req[1]);
            MPI_Irecv(curLayer, N_X * N_Y, MPI_DOUBLE,
                      rank - 1, 1, MPI_COMM_WORLD, &req[0]);
        }
        if (rank != size - 1) {
            MPI_Isend(curLayer + N_X * N_Y * layerSize, N_X * N_Y, MPI_DOUBLE,
                      rank + 1, 1, MPI_COMM_WORLD, &req[3]);
            MPI_Irecv(curLayer + N_X * N_Y * (layerSize + 1), N_X * N_Y, MPI_DOUBLE,
                      rank + 1, 1, MPI_COMM_WORLD, &req[2]);
        }

        for (int i = 2; i < layerSize; i++) {
            tmpMaxDelta = calculateLayer(layerZCoord, i, prevLayer, curLayer);
            procMaxDelta = fmax(procMaxDelta, tmpMaxDelta);
        }

        if (rank != size - 1) {
            MPI_Wait(&req[2], MPI_STATUS_IGNORE);
            MPI_Wait(&req[3], MPI_STATUS_IGNORE);
        }
        if (rank != 0) {
            MPI_Wait(&req[0], MPI_STATUS_IGNORE);
            MPI_Wait(&req[1], MPI_STATUS_IGNORE);
        }
        tmpMaxDelta = calculateLayer(layerZCoord, 1, prevLayer, curLayer);
        procMaxDelta = fmax(procMaxDelta, tmpMaxDelta);

        tmpMaxDelta = calculateLayer(layerZCoord, layerSize, prevLayer, curLayer);
        procMaxDelta = fmax(procMaxDelta, tmpMaxDelta);

        memcpy(prevLayer, curLayer, usedLayerSize * sizeof(double));
        MPI_Allreduce(&procMaxDelta, &globalMaxDelta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    }

    double end = MPI_Wtime();
    if (RANK_ROOT == rank)
        area = (double *)malloc(N_X * N_Y * N_Z * sizeof(double));

    MPI_Gather(prevLayer + N_X * N_Y, layerSize * N_X * N_Y, MPI_DOUBLE, area, layerSize * N_X * N_Y, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (RANK_ROOT == rank) {
        printf("Time taken: %10.5f\n", end - start);
        printf("Max delta: %10.5f\n", calculateDelta(area));
        if (area != NULL)
            free(area);
    }

    free(curLayer);
    free(prevLayer);
    MPI_Finalize();
    return 0;
}
