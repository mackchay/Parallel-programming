#!/bin/bash
set -x

mpicc -std=gnu99 -O0 -Wpedantic -Wall -Werror -Wextra -lm -o lab3 main.c