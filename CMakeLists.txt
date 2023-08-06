cmake_minimum_required(VERSION 3.10.0)
project(Clearn VERSION 0.1.0 LANGUAGES C CXX)

# include(CTest)
# enable_testing()

# Source files
add_executable(Clearn main.cpp utils.h)

# headers file
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_SOURCE_DIR D:/cpp_lib)
set(Vulkan_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/vulkan/Include)
set(Vulkan_LIBS ${CMAKE_SOURCE_DIR}/vulkan/Lib/*.lib)

set(NCNN_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/ncnn/x64/include/ncnn)
set(NCNN_LIBS ${CMAKE_SOURCE_DIR}/ncnn/x64/lib/*.lib)
# set(Vulkan_INCLUDE_DIR )

#======== Lib and dependency========
# find_package(OpenCV REQUIRED core imgproc)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( Clearn ${OpenCV_LIBS} )

find_package(Vulkan REQUIRED)
include_directories( ${Vulkan_INCLUDE_DIR} )
target_link_libraries(Clearn ${Vulkan_LIBS} )

find_package(NCNN REQUIRED)
include_directories( ${NCNN_INCLUDE_DIRS} )
target_link_libraries( Clearn ${NCNN_LIBS} )
#===================================

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)

message(CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
message(OpenCV_LIBS="${OpenCV_LIBS}")
message(NCNN_INCLUDE_DIRS="${NCNN_INCLUDE_DIRS}")
message(NCNN_LIBS="${NCNN_LIBS}")
