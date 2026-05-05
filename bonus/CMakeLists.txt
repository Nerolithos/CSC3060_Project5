cmake_minimum_required(VERSION 3.20)
project(csc3060_project5_bonus_notes LANGUAGES CXX)

add_library(bonus_notes OBJECT bonus_notes.cpp)
target_compile_features(bonus_notes PRIVATE cxx_std_23)

find_package(OpenMP)

add_library(matmul_bonus_openmp OBJECT matmul_bonus_openmp.cpp)
target_compile_features(matmul_bonus_openmp PRIVATE cxx_std_23)
target_include_directories(matmul_bonus_openmp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(matmul_bonus_openmp PRIVATE OpenMP::OpenMP_CXX)
endif()cmake_minimum_required(VERSION 3.20)
project(csc3060_project5_bonus_notes LANGUAGES CXX)

add_library(bonus_notes OBJECT bonus_notes.cpp)
target_compile_features(bonus_notes PRIVATE cxx_std_23)

find_package(OpenMP)

add_library(matmul_bonus_openmp OBJECT matmul_bonus_openmp.cpp)
target_compile_features(matmul_bonus_openmp PRIVATE cxx_std_23)
target_include_directories(matmul_bonus_openmp PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(OpenMP_CXX_FOUND)
	target_link_libraries(matmul_bonus_openmp PRIVATE OpenMP::OpenMP_CXX)
endif()
