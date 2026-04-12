cmake_minimum_required(VERSION 3.12)
project(lab_3)
set(CMAKE_CXX_STANDARD 20)
add_executable(lab_3 CSquareMatrix.cpp)

find_package(MPI REQUIRED)
target_link_libraries(lab_3 PRIVATE MPI::MPI_CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

if(MSVC)
    target_compile_options(lab_3 PRIVATE /EHsc)
    target_link_options(lab_3 PRIVATE /STACK:67108864)
endif()