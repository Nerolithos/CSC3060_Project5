cmake_minimum_required(VERSION 3.20)
project(csc3060_project5_bonus LANGUAGES CXX)

add_library(matmul_bonus_threaded OBJECT matmul_bonus_threaded.cpp)
target_compile_features(matmul_bonus_threaded PRIVATE cxx_std_23)
target_include_directories(matmul_bonus_threaded PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
	add_library(bitwise_bonus_simd OBJECT bitwise_bonus_simd.cpp)
	target_compile_features(bitwise_bonus_simd PRIVATE cxx_std_23)
	target_include_directories(bitwise_bonus_simd PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/../include
	)

	if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
		target_compile_options(bitwise_bonus_simd PRIVATE -mavx2)
	endif()
else()
	message(STATUS "Skipping bitwise_bonus_simd: AVX2 requires an x86 target")
endif()
