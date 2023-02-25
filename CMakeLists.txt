cmake_minimum_required(VERSION 3.21)
project(paralelism C)

set(CMAKE_C_STANDARD 23)

add_executable(paralelism main.c double.c)
