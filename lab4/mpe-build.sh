#!/bin/bash
set -x

mpecc -mpilog -std=gnu99 -O2 -Wpedantic -Wall -Werror -Wextra -lm -o lab4 main.c
