cmake_minimum_required(VERSION 3.8)
project(ML_CPP)

set(CMAKE_CXX_STANDARD 11)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

include_directories(
    ${Python3_INCLUDE_DIRS}
    matplotlib-cpp
    linear_regression
    svm
    common
    "D:\\Users\\44225\\anaconda3\\Lib\\site-packages\\numpy\\core\\include"
)

file(GLOB_RECURSE SOURCES "linear_regression/*.cpp" "common/*.cpp" "svm/*.cpp")

add_executable(main main.cpp ${SOURCES})
target_link_libraries(main ${Python3_LIBRARIES})