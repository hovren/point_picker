cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(pcl_point_pick)

find_package(PCL REQUIRED)
#find_package(PCL 1.7 REQUIRED COMPONENTS visualization registration common io console)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pcl_point_pick pcl_point_pick.cpp)
target_link_libraries (pcl_point_pick ${PCL_LIBRARIES})
