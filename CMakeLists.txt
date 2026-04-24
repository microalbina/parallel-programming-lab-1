cmake_minimum_required(VERSION 3.12)
project(lab_4)
set(CMAKE_CXX_STANDARD 20)
add_executable(lab_4 CSquareMatrix.cpp)

find_package(MPI REQUIRED)
target_link_libraries(lab_4 PRIVATE MPI::MPI_CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

if(MSVC)
    target_compile_options(lab_4 PRIVATE /EHsc)
    target_link_options(lab_4 PRIVATE /STACK:67108864)
endif()