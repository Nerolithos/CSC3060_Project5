cmake_minimum_required(VERSION 3.20)
project(csc3060_project5_bonus LANGUAGES CXX)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    add_executable(run_bonus run_bonus.cpp image_proc.cpp)
    target_compile_features(run_bonus PRIVATE cxx_std_23)
    target_compile_options(run_bonus PRIVATE -O2 -ffast-math -Wall -g -fno-omit-frame-pointer)
    target_include_directories(run_bonus PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
    target_link_libraries(run_bonus PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "Bonus target 'run_bonus' configured with OpenMP.")
else()
    message(WARNING "OpenMP not found; bonus target 'run_bonus' will not be built.")
endif()
