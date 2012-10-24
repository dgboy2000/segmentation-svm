ADD_LIBRARY(fastPD SHARED
  src/maxflow.cpp
  src/LinkedBlockList.cpp
  src/graph.cpp
  src/CV_Fast_PD.h
  src/CV_Fast_PD_costfunction.h
  fastPD.cpp
  fastPD_callback.cpp
  fastPD_potts.cpp
  )

SET_TARGET_PROPERTIES(fastPD PROPERTIES LINKER_LANGUAGE CXX)

# ADD_LIBRARY(fastPD_callback SHARED
  # src/maxflow.cpp
  # src/LinkedBlockList.cpp
  # src/graph.cpp
 
  # )

# SET_TARGET_PROPERTIES(fastPD_callback PROPERTIES LINKER_LANGUAGE CXX) 

SET(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib/")

