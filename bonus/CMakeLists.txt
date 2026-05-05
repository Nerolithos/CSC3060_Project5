cmake_minimum_required(VERSION 3.20)
project(csc3060_project5_bonus_notes LANGUAGES CXX)

add_library(bonus_notes OBJECT bonus_notes.cpp)
target_compile_features(bonus_notes PRIVATE cxx_std_23)
