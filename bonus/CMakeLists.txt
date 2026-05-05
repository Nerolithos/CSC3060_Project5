cmake_minimum_required(VERSION 3.20)
project(csc3060_project5_bonus LANGUAGES CXX)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    add_library(image_proc_bonus_ported OBJECT image_proc.cpp)
    target_compile_features(image_proc_bonus_ported PRIVATE cxx_std_23)
    target_include_directories(image_proc_bonus_ported PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
    target_link_libraries(image_proc_bonus_ported PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found; image_proc bonus target will not be built.")
endif()
