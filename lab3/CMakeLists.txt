cmake_minimum_required(VERSION 3.5.2)
project(lab3 C)

set(CMAKE_C_STANDARD 99)

include_directories(/usr/include/x86_64-linux-gnu/mpich)
set(CMAKE_C_COMPILER "/usr/bin/mpicc")
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")

add_executable(lab3 main.c)
target_link_libraries(lab3 m)