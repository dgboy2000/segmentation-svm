
ADD_LIBRARY(ctrw SHARED
    src/MRFEnergy.cpp
    src/treeProbabilities.cpp
    src/ordering.cpp
    src/minimize.cpp
  )

SET_TARGET_PROPERTIES(ctrw PROPERTIES LINKER_LANGUAGE CXX)  


ADD_LIBRARY(trw SHARED
  trw_binary.cpp
  trw_general.cpp
  trw_potts.cpp
  trw_potts_wlabels.cpp
) 
TARGET_LINK_LIBRARIES(trw
  ctrw
)

SET_TARGET_PROPERTIES(trw PROPERTIES LINKER_LANGUAGE CXX)

SET(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib/")


