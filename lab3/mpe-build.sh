#!/bin/bash
set -x

mpecc -mpilog -std=gnu99 -O0 -Wpedantic -Wall -Werror -Wextra -lm -o lab3 main.c
