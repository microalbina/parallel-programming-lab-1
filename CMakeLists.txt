cmake_minimum_required(VERSION 3.12)
project(lab_1)
set(CMAKE_CXX_STANDARD 20)
add_executable(lab_1 CSquareMatrix.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

if(MSVC)
    target_link_options(lab_1 PRIVATE /STACK:67108864)
endif()