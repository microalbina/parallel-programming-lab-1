cmake_minimum_required(VERSION 3.12)
project(lab_2)
set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)
add_executable(lab_2 CSquareMatrix.cpp)
target_link_libraries(lab_2 PRIVATE OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

if(MSVC)
    target_link_options(lab_2 PRIVATE /STACK:67108864)
endif()