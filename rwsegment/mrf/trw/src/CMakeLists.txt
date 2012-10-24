PROJECT(TRWCPP)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)


ADD_LIBRARY(testtrw SHARED
    MRFEnergy.cpp
    treeProbabilities.cpp
    ordering.cpp
    minimize.cpp
  )

SET_TARGET_PROPERTIES(testtrw PROPERTIES LINKER_LANGUAGE CXX)  


