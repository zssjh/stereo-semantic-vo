# project
cmake_minimum_required (VERSION 2.6)
project (libelas)

# directories
set (LIBELAS_SRC_DIR src)
set(SRCS_libelas
  src/descriptor.cpp
  src/elas.cpp      
  src/filter.cpp
  src/matrix.cpp
  src/triangle.cpp)

# include directory
include_directories("${LIBELAS_SRC_DIR}")

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")

# sources
FILE(GLOB LIBELAS_SRC_FILES "src/*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_library(libelas SHARED ${SRCS_libelas})

# build demo program
# add_executable(elas ${LIBELAS_SRC_FILES})

