#include <iostream>
#include <pthread.h>
#include <cmath>
#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr int L = 1000;
constexpr int LISTS_COUNT = 10;
constexpr int TASK_COUNT = 10;
constexpr int MIN_TASKS_TO_SHARE = 2;
constexpr int EXECUTOR_FINISHED_WORK = -1;
constexpr int ASKING_TASK = 888;
constexpr int SENDING_TASKS = 656;
constexpr int SENDING_TASK_COUNT = 787;
constexpr int NO_TASKS_TO_SHARE = -2;
constexpr int COS_CONST = 10;
constexpr double PERCENT = 0.5;
constexpr double RANK_ROOT = 0;

struct GlobalElements {
    int *tasks = nullptr;
    double summaryDisBalance = 0;
    bool finishedExecution = false;

    int remainingTasks = 0;
    int executedTasks = 0;
    double globalRes = 0;
    double totalGlobalResult = 0;
};

GlobalElements globalElements;
pthread_mutex_t mutex;

void initTasks(int iter) {
    int processRank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    globalElements.tasks = new int[TASK_COUNT];
    globalElements.remainingTasks = TASK_COUNT;
    for (int i = 0; i < globalElements.remainingTasks; i++) {
        globalElements.tasks[i] = abs(50 - i % 100) * abs(processRank - (iter % processCount)) * L;
    }
}

void executeTasks() {
    int index = 0;
    while (true) {
        pthread_mutex_lock(&mutex);
        if (globalElements.remainingTasks > 0) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        int weight = globalElements.tasks[index];
        globalElements.remainingTasks--;
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

void finishWork() {
    pthread_mutex_lock(&mutex);
    globalElements.finishedExecution = true;
    pthread_mutex_unlock(&mutex);
    int processRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    int signal = EXECUTOR_FINISHED_WORK;
    MPI_Send(&signal, 1, MPI_INT, processRank,
             SENDING_TASK_COUNT, MPI_COMM_WORLD);
}

bool askForTasks() {
    int processRank, processCount;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    for (int process = 0; process < processCount; process++) {
        int receivedTasksCount;
        if (process == processRank) {
            continue;
        }
        MPI_Send(&processRank, 1, MPI_INT, process,
                 ASKING_TASK, MPI_COMM_WORLD);
        MPI_Recv(&receivedTasksCount, 1, MPI_INT, process,
                 SENDING_TASK_COUNT, MPI_COMM_WORLD, &status);
        if (receivedTasksCount == NO_TASKS_TO_SHARE) {
            continue;
        }
        memset(globalElements.tasks, 0, TASK_COUNT * sizeof(int));
        MPI_Recv(globalElements.tasks, receivedTasksCount, MPI_INT, process,
                 SENDING_TASKS, MPI_COMM_WORLD, &status);
        pthread_mutex_lock(&mutex);
        globalElements.remainingTasks = receivedTasksCount;
        pthread_mutex_unlock(&mutex);
        executeTasks();
    }
    return false;
}

void executorThread() {
    double startTime, finishTime, iterDuration, shortestIter, longestIter, currentDisBalance;
    double localResult, globalDurationResult;
    int processRank, totalExecutedTasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    for (int iter = 0; iter < LISTS_COUNT; iter++) {
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        initTasks(iter);
        executeTasks();
        askForTasks();
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
        globalElements.summaryDisBalance += currentDisBalance;
        globalElements.totalGlobalResult += globalDurationResult;
//        std::cout << "Sum of square roots on process " << processRank << " is " <<
//                  globalElements.globalRes << ". Time taken: " <<
//                  iterDuration << std::endl;
        if (processRank == RANK_ROOT) {
            std::cout << "Total executed tasks: " << totalExecutedTasks << std::endl;
            std::cout << "Max time difference: " << longestIter - shortestIter << std::endl;
            std::cout << "DisBalance rate is " << currentDisBalance * 100 << "%" << std::endl;
            std::cout << "Sum of square roots on all processes in iteration " << iter << " is "
                      << globalElements.summaryDisBalance
                      << std::endl;
            std::cout << std::endl;
        }
    }
    if (processRank == RANK_ROOT) {
        std::cout << "Global result: " << globalElements.totalGlobalResult << std::endl;
    }
    finishWork();
}

void *receiverThread(void *data) {

    int processAsking, additionalTasks;
    int taskCountLocal = TASK_COUNT;
    MPI_Status status;
    while (true) {
        pthread_mutex_lock(&mutex);
        if (globalElements.finishedExecution) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        pthread_mutex_unlock(&mutex);
        MPI_Recv(&processAsking, 1, MPI_INT, MPI_ANY_SOURCE,
                 ASKING_TASK, MPI_COMM_WORLD, &status);
        if (processAsking == EXECUTOR_FINISHED_WORK) {
            break;
        }
        if (taskCountLocal >= MIN_TASKS_TO_SHARE) {
            additionalTasks = static_cast<int>(taskCountLocal * PERCENT);
            taskCountLocal -= additionalTasks;
            pthread_mutex_lock(&mutex);
            globalElements.remainingTasks -= additionalTasks;
            pthread_mutex_unlock(&mutex);
            MPI_Send(&additionalTasks, 1, MPI_INT, processAsking,
                     SENDING_TASK_COUNT, MPI_COMM_WORLD);
            MPI_Send(&globalElements.tasks[taskCountLocal], additionalTasks,
                     MPI_INT, processAsking,
                     SENDING_TASKS, MPI_COMM_WORLD);
        } else {
            additionalTasks = NO_TASKS_TO_SHARE;
            MPI_Send(&additionalTasks, 1, MPI_INT, processAsking,
                     SENDING_TASK_COUNT, MPI_COMM_WORLD);
        }
    }
    pthread_exit(nullptr);
}

int main() {
    int threadError;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &threadError);
    if (threadError != MPI_THREAD_MULTIPLE) {
        MPI_Finalize();
        return 1;
    }

    pthread_t pthread;
    pthread_mutex_init(&mutex, nullptr);
    pthread_create(&pthread, nullptr, receiverThread, nullptr);
    executorThread();
    pthread_join(pthread, nullptr);
    pthread_mutex_destroy(&mutex);
    delete[] globalElements.tasks;
    MPI_Finalize();
    return 0;
}