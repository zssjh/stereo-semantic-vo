cmake_minimum_required(VERSION 2.8)
project(MB)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV)

set(SOURCE_FILES
        ctmf.c
        ctmf.h
        MSA.cpp
        MSA.h)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_library(MB SHARED ${SOURCE_FILES})

#add_executable(MB
#${SOURCE_FILES}
#main.cpp)
#target_link_libraries(MB ${OpenCV_LIBS})
