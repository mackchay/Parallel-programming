#include <iostream>
#include <pthread.h>
#include <cmath>
#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <vector>

#define L 1000
#define LISTS_COUNT 10
#define TASK_COUNT 10
#define MIN_TASKS_TO_SHARE 2
#define EXECUTOR_FINISHED_WORK -1
#define SENDING_TASKS 656
#define SENDING_TASK_COUNT 787
#define NO_TASKS_TO_SHARE -2

struct GlobalElements {
    int * tasks = nullptr;
    double summaryDisBalance = 0;
    bool finishedExecution = false;

    int remainingTasks = 0;
    int executedTasks = 0;
    int additionalTasks = 0;
};

pthread_mutex_t mutex;
GlobalElements globalElements;
int globalRes;

void initializeTaskSet(int * taskSet, int taskCount, int iterCounter) {
    int processRank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    for (int i = 0; i < taskCount; i++) {
        taskSet[i] = abs(50 - i%100) * abs(processRank -
                (iterCounter % processCount)) * L;
    }
}

void executeTaskSet(int * taskSet) {
    int i = 0;
    while (true) {
        pthread_mutex_lock(&mutex);
        if (i == globalElements.remainingTasks) {
            pthread_mutex_unlock(&mutex);
            break;
        }

        globalElements.executedTasks++;
        for (int j = 0; j < taskSet[i]; j++) {
            globalRes += cos(0.001488);
        }
        pthread_mutex_unlock(&mutex);
        i++;
    }
    globalElements.remainingTasks = 0;
}

void * AddTask() {
    int threadResponse;
    int processRank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    for (int procIdx = 0; procIdx < processCount; procIdx++) {
        if (procIdx == processRank)
            continue;
        MPI_Send(&processRank, 1, MPI_INT, procIdx, 888, MPI_COMM_WORLD);
        MPI_Recv(&threadResponse, 1, MPI_INT, procIdx, SENDING_TASK_COUNT,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (threadResponse == NO_TASKS_TO_SHARE)
            continue;
        std::cout << "Process " << processRank << " received " << threadResponse << " tasks from process "
        << procIdx;
        globalElements.additionalTasks = threadResponse;
        memset(globalElements.tasks, 0, TASK_COUNT);
        MPI_Recv(globalElements.tasks, globalElements.additionalTasks,
                 MPI_INT, procIdx, SENDING_TASKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pthread_mutex_lock(&mutex);
        globalElements.remainingTasks = globalElements.additionalTasks;
        pthread_mutex_unlock(&mutex);
        executeTaskSet(globalElements.tasks);
    }
    //pthread_exit(nullptr);
}

void* executorStartRoutine() {
    int processRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    globalElements.tasks = new int[TASK_COUNT];
    double StartTime, FinishTime, IterationDuration, ShortestIteration, LongestIteration;

    for (int i = 0; i < LISTS_COUNT; i++) {
        StartTime = MPI_Wtime();
        initializeTaskSet(globalElements.tasks, TASK_COUNT, i);
        pthread_mutex_lock(&mutex);
        globalElements.executedTasks = 0;
        globalElements.remainingTasks = TASK_COUNT;
        globalElements.additionalTasks = 0;
        pthread_mutex_unlock(&mutex);
        executeTaskSet(globalElements.tasks);
        AddTask();
        FinishTime = MPI_Wtime();
        IterationDuration = FinishTime - StartTime;
        MPI_Allreduce(&IterationDuration, &LongestIteration, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&IterationDuration, &ShortestIteration, 1, MPI_DOUBLE,
                      MPI_MIN, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        pthread_mutex_lock(&mutex);
        globalElements.summaryDisBalance += (LongestIteration - ShortestIteration) / LongestIteration;
        pthread_mutex_unlock(&mutex);

    }
    pthread_mutex_lock(&mutex);
    globalElements.finishedExecution = true;
    pthread_mutex_unlock(&mutex);
    int Signal = EXECUTOR_FINISHED_WORK;
    MPI_Send(&Signal, 1, MPI_INT, processRank, 888, MPI_COMM_WORLD);
    pthread_exit(nullptr);
}

void* receiverStartRoutine(void *glp) {
    int processRank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    int askingProcRank, Answer, PendingMessage;
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);
    while (!globalElements.finishedExecution) {
        MPI_Recv(&PendingMessage, 1, MPI_INT, MPI_ANY_SOURCE, 888, MPI_COMM_WORLD,
                 &status);
        askingProcRank = PendingMessage;
        pthread_mutex_lock(&mutex);
        if (globalElements.remainingTasks >= MIN_TASKS_TO_SHARE) {
            Answer = globalElements.remainingTasks / (processCount * 2);
            pthread_mutex_lock(&mutex);
            globalElements.remainingTasks = globalElements.remainingTasks /
                    (processCount * 2);
            MPI_Send(&Answer, 1, MPI_INT, askingProcRank,
                     SENDING_TASK_COUNT,
                     MPI_COMM_WORLD);
            MPI_Send(&globalElements.tasks[TASK_COUNT - Answer],
                     Answer, MPI_INT, askingProcRank,
                     SENDING_TASKS, MPI_COMM_WORLD);
            std::cout << "Process " << processRank << " shared with "
            << askingProcRank << " his" <<
            TASK_COUNT - Answer << " tasks.";
            pthread_mutex_unlock(&mutex);
        } else if (askingProcRank != processRank) {
            pthread_mutex_unlock(&mutex);
            Answer = NO_TASKS_TO_SHARE;
            MPI_Send(&Answer, 1, MPI_INT,
                     askingProcRank, SENDING_TASK_COUNT,
                     MPI_COMM_WORLD);
        }
    }
}


int main(int argc, char* argv[]) {
    int threadSupport;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSupport);
    if(threadSupport != MPI_THREAD_MULTIPLE) {
        std::cout << "Error" << std::endl;
        MPI_Finalize();
        return -1;
    }

    int processRank, processCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    pthread_mutex_init(&mutex, nullptr);
    pthread_attr_t threadAttributes;
    pthread_t thread;

    double start = MPI_Wtime();
    pthread_attr_init(&threadAttributes);
    pthread_attr_setdetachstate(&threadAttributes, PTHREAD_CREATE_JOINABLE);
    pthread_create(&thread, &threadAttributes, receiverStartRoutine,
                   nullptr);
    executorStartRoutine();
    pthread_join(thread, nullptr);
    pthread_attr_destroy(&threadAttributes);
    pthread_mutex_destroy(&mutex);

    if (processRank == 0) {
        std::cout << "Summary dis balance: " << globalElements.summaryDisBalance / (LISTS_COUNT) * 100 << "%" << std::endl;
        std::cout << "time taken: " << MPI_Wtime() - start << std::endl;
    }

    MPI_Finalize();
    return 0;
}