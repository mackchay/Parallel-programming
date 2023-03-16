cmake_minimum_required(VERSION 3.5.2)
project(lab1 C)

set(CMAKE_C_STANDARD 99)

include_directories(/usr/include/x86_64-linux-gnu/mpich)
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Werror -Wall -Wextra -Wpedantic")

add_executable(lab2 main.c)
target_link_libraries(lab2 m)