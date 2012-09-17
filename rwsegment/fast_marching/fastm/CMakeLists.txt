PROJECT(FASTM)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

ADD_LIBRARY( fastm SHARED
    fast_marching_3d_general.cpp
  )
  
set_target_properties(fastm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/libs)
