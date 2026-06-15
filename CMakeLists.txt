cmake_minimum_required(VERSION 3.22)
project(DigraChurn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# same spirit as DIGRA's build (O3 + openmp + vectorize), but DO NOT force
# -mavx512f: if the CPU lacks AVX-512, forcing it yields SIGILL ("illegal
# instruction") at runtime. -march=native enables exactly what THIS cpu has.
SET(CMAKE_CXX_FLAGS "-O3 -march=native -lrt -DHAVE_CXX0X -fpic -w -fopenmp -ftree-vectorize")

option(ENABLE_SSE "Enable SSE support" ON)

# put all built executables in ./bin so they don't clutter the repo root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# build the churn driver against DIGRA's hnswlib + TreeHNSW.hpp.
# Place this CMakeLists + digra_churn.cpp inside the DIGRA repo root
# (next to TreeHNSW.hpp, utils.hpp, and the hnswlib/ dir).
add_executable(digra_churn
        digra_churn.cpp
        hnswlib/bruteforce.h
        hnswlib/hnswalg.h
        hnswlib/hnswlib.h
        hnswlib/space_ip.h
        hnswlib/space_l2.h
        hnswlib/stop_condition.h
        hnswlib/visited_list_pool.h
        utils.hpp
        TreeHNSW.hpp
)

# vanilla hnswlib churn baseline (no B-tree, pure HNSW add/markDelete)
add_executable(hnsw_churn
        hnsw_churn.cpp
        hnswlib/bruteforce.h
        hnswlib/hnswalg.h
        hnswlib/hnswlib.h
        hnswlib/space_ip.h
        hnswlib/space_l2.h
        hnswlib/stop_condition.h
        hnswlib/visited_list_pool.h
)