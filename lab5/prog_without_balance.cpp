#include <iostream>
#include <pthread.h>
#include <cmath>
#include <mpi.h>
#include <cstdlib>
#include <vector>

constexpr int L = 1000;
constexpr int LISTS_COUNT = 10;
constexpr int TASK_COUNT = 10;
constexpr int COS_CONST = 10;
constexpr double RANK_ROOT = 0;

typedef struct {
    int *tasks;
    double summaryDisBalance;
    bool finishedExecution;

    int remainingTasks;
    int executedTasks;
    double globalRes;
    double totalGlobalResult;
    double globalDuration;
} GlobalElements;

GlobalElements globalElements;
pthread_mutex_t mutex;

void initTasks(int iter) {
    int processRank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    pthread_mutex_lock(&mutex);
    globalElements.remainingTasks = TASK_COUNT;
    for (int i = 0; i < globalElements.remainingTasks; i++) {
        globalElements.tasks[i] = abs(50 - i % 100) * abs(processRank - (iter % processCount)) * L;
    }
    pthread_mutex_unlock(&mutex);
}

void executeTasks() {
    int index = 0;
    while (true) {
        pthread_mutex_lock(&mutex);
        if (globalElements.remainingTasks <= 0) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        int weight = globalElements.tasks[index];
        globalElements.remainingTasks--;
        globalElements.executedTasks++;
        pthread_mutex_unlock(&mutex);
        index++;
        for (int i = 0; i < weight; i++) {
            globalElements.globalRes += cos(i * COS_CONST);
        }
    }
    pthread_mutex_lock(&mutex);
    globalElements.remainingTasks = 0;
    pthread_mutex_unlock(&mutex);
}

void initGlobalData() {
    pthread_mutex_lock(&mutex);
    globalElements.tasks = new int[TASK_COUNT];
    globalElements.totalGlobalResult = 0;
    globalElements.executedTasks = 0;
    globalElements.globalDuration = 0;
    globalElements.finishedExecution = false;
    globalElements.globalRes = 0;
    globalElements.summaryDisBalance = 0;
    pthread_mutex_unlock(&mutex);
}

void executor() {
    double startTime, finishTime, iterDuration, shortestIter, longestIter, currentDisBalance;
    double localResult, globalDurationResult;
    int processRank, totalExecutedTasks;
    initGlobalData();
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    for (int iter = 0; iter < LISTS_COUNT; iter++) {
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        initTasks(iter);
        executeTasks();
        finishTime = MPI_Wtime();
        iterDuration = finishTime - startTime;

        MPI_Allreduce(&iterDuration, &longestIter, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iterDuration, &shortestIter, 1,
                      MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&globalElements.executedTasks, &totalExecutedTasks, 1,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localResult, &globalDurationResult, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        currentDisBalance = (longestIter - shortestIter) / longestIter;
        pthread_mutex_lock(&mutex);
        globalElements.summaryDisBalance += currentDisBalance;
        globalElements.totalGlobalResult += globalDurationResult;
        globalElements.globalDuration += iterDuration;
        pthread_mutex_unlock(&mutex);
//        std::cout << "Sum of square roots on process " << processRank << " is " <<
//                  globalElements.globalRes << ". Time taken: " <<
//                  iterDuration << std::endl;
        if (processRank == RANK_ROOT) {
            std::cout << "Total executed tasks: " << totalExecutedTasks << std::endl;
            std::cout << "Max time difference: " << longestIter - shortestIter << std::endl;
            std::cout << "DisBalance rate is " << currentDisBalance * 100 << "%" << std::endl;
            std::cout << "Cos function value on all processes in iteration " << iter << " is "
                      << globalElements.totalGlobalResult
                      << std::endl;
            std::cout << std::endl;
        }
    }

    if (processRank == RANK_ROOT) {
        std::cout << "Global result: " << globalElements.totalGlobalResult << std::endl;
        std::cout << "Summary disbalance:" << globalElements.summaryDisBalance / (TASK_COUNT) * 100 << "%" << std::endl;
        std::cout << "Time taken: " << globalElements.globalDuration << std::endl;
    }
}

int main() {
    MPI_Init(nullptr, nullptr);

    executor();
    delete[] globalElements.tasks;
    MPI_Finalize();
    return 0;
}
