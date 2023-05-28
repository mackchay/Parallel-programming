#include <iostream>
#include <pthread.h>
#include <cmath>
#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr int L = 1000;
constexpr int LISTS_COUNT = 10;
constexpr int TASK_COUNT = 1000;
constexpr int MIN_TASKS_TO_SHARE = 2;
constexpr int EXECUTOR_FINISHED_WORK = -1;
constexpr int ASKING_TASK = 888;
constexpr int SENDING_TASKS = 656;
constexpr int SENDING_TASK_COUNT = 787;
constexpr int NO_TASKS_TO_SHARE = -2;
constexpr int COS_CONST = 10;
constexpr double PERCENT = static_cast<double>(50) / static_cast<double>(100);

struct GlobalElements {
    int *tasks = nullptr;
    double summaryDisBalance = 0;
    bool finishedExecution = false;

    int remainingTasks = 0;
    int executedTasks = 0;
    int additionalTasks = 0;
};

pthread_mutex_t mutex;
GlobalElements globalElements;
int globalRes;

void initTasks(int iter) {
    int processRank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    pthread_mutex_lock(&mutex);
    globalElements.tasks = new int[TASK_COUNT];
    globalElements.remainingTasks = TASK_COUNT;
    for (int i = 0; i < globalElements.remainingTasks; i++) {
        globalElements.tasks[i] = abs(50 - i % 100) * abs(processRank - (iter % processCount)) * L;
    }
    pthread_mutex_unlock(&mutex);
}

void executeTasks() {
    pthread_mutex_lock(&mutex);
    while (globalElements.remainingTasks > 0) {
        pthread_mutex_unlock(&mutex);
        for (int i = 0; i < globalElements.tasks[globalElements.remainingTasks - 1]; i++) {
            globalRes += cos(i * COS_CONST);
        }
        pthread_mutex_lock(&mutex);
        globalElements.remainingTasks--;
    }
    pthread_mutex_unlock(&mutex);
}

void askForTasks() {
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
        if (receivedTasksCount != NO_TASKS_TO_SHARE) {
            MPI_Recv(globalElements.tasks, receivedTasksCount, MPI_INT, process,
                     SENDING_TASKS, MPI_COMM_WORLD, &status);
            pthread_mutex_lock(&mutex);
            std::cout << "Process " << processRank << " got new tasks." << std::endl;
            globalElements.remainingTasks = receivedTasksCount;
            pthread_mutex_unlock(&mutex);
            return;
        }
        pthread_mutex_unlock(&mutex);
    }
    MPI_Send(&EXECUTOR_FINISHED_WORK, 1, MPI_INT, processRank,
             SENDING_TASK_COUNT, MPI_COMM_WORLD);
    std::cout << "hello" << std::endl;
}

void executorThread() {
    for (int iter = 0; iter < LISTS_COUNT; iter++) {
        initTasks(iter);
        executeTasks();
        askForTasks();
        executeTasks();
    }
}

void *receiverThread(void *data) {
    int processAsking;
    MPI_Status status;
    while (true) {
        MPI_Recv(&processAsking, 1, MPI_INT, MPI_ANY_SOURCE,
                 ASKING_TASK, MPI_COMM_WORLD, &status);
        if (processAsking == EXECUTOR_FINISHED_WORK) {
            break;
        }
        pthread_mutex_lock(&mutex);
        if (globalElements.remainingTasks >= MIN_TASKS_TO_SHARE) {
            globalElements.additionalTasks = static_cast<int>(globalElements.remainingTasks * PERCENT);
            globalElements.remainingTasks -= globalElements.additionalTasks;
            MPI_Send(&globalElements.additionalTasks, 1, MPI_INT, processAsking,
                     SENDING_TASK_COUNT, MPI_COMM_WORLD);
            MPI_Send(&globalElements.tasks[globalElements.remainingTasks], globalElements.additionalTasks,
                     MPI_INT, processAsking,
                     SENDING_TASKS, MPI_COMM_WORLD);
            pthread_mutex_unlock(&mutex);
        }
        else {
            pthread_mutex_unlock(&mutex);
            MPI_Send(&NO_TASKS_TO_SHARE, 1, MPI_INT, processAsking,
                     SENDING_TASK_COUNT, MPI_COMM_WORLD);
        }
    }
    pthread_exit(nullptr);
}

int main() {
    MPI_Init(nullptr, nullptr);
    pthread_attr_t pthreadAttr;
    pthread_attr_init(&pthreadAttr);
    pthread_t pthread;
    pthread_create(&pthread, &pthreadAttr, receiverThread, nullptr);
    executorThread();
    MPI_Finalize();
    return 0;
}