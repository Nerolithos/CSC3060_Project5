cmake_minimum_required(VERSION 3.20)
project(csc3060_project5_bonus_notes LANGUAGES CXX)

add_library(bonus_notes OBJECT bonus_notes.cpp)
target_compile_features(bonus_notes PRIVATE cxx_std_23)

add_library(bitwise_bonus_simd OBJECT bitwise_bonus_simd.cpp)
target_compile_features(bitwise_bonus_simd PRIVATE cxx_std_23)
target_include_directories(bitwise_bonus_simd PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
	target_compile_options(bitwise_bonus_simd PRIVATE -mavx2)
endif()
