cmake_minimum_required(VERSION 2.8)
project(stereo_vo)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


include(FindCUDA)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

find_package(OpenCV QUIET)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
find_package(CUDA 8.0 REQUIRED)

set(DARKNET_PATH ${PROJECT_SOURCE_DIR}/Thirdparty/darknet)
add_definitions(-DDARKNET_FILE_PATH="${DARKNET_PATH}")

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${DARKNET_PATH}/src
        /usr/local/cuda/include
)
set(CUDA_INCLUDE_DIR /usr/local/cuda/include)
set(CUDNN_LIBRARY_DIR /usr/local/cuda-8.0/targets/x86_64-linux/lib)

add_library(${PROJECT_NAME} SHARED
        src/Tracking.cc
        include/frame.h
        src/frame.cc
        include/mappoint.h
        src/mappoint.cc
        include/map.h
        src/map.cc
        include/pnpmatch.h
        src/pnpmatch.cc
        include/convert.h
        src/convert.cc
        src/Optimizer.cc
        include/Optimizer.h
        include/view.h
        src/view.cc include/semantic.h src/semantic.cc)

add_library(cuda INTERFACE)
set_target_properties(cuda PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
        INTERFACE_LINK_LIBRARIES "${CUDA_LIBRARIES};${CUDA_curand_LIBRARY};${CUDA_CUFFT_LIBRARIES};${CUDA_CUBLAS_LIBRARIES}"
        )

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/libelas/lib/liblibelas.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/darknet/build/libYOLOv3SE.so
        /usr/local/cuda-8.0/lib64/libcudnn.so
        cuda
        )

# Build examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Stereo)
add_executable(stereo_kitti
        main.cpp)
target_link_libraries(stereo_kitti ${PROJECT_NAME})



