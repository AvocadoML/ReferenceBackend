cmake_minimum_required(VERSION 3.18)
project(ReferenceBackend VERSION 1.0.0 DESCRIPTION "Reference backend implementation" LANGUAGES CXX)

option(BUILD_TESTING "Build unit tests?" OFF)
option(BUILD_SHARED_LIBS "Build shared library?" OFF)
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Choose debug postfix" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type" FORCE)
set(PATH_TO_MAIN_LIB "${PROJECT_SOURCE_DIR}/../Avocado" CACHE STRING "Path to the main Avocado project")

set(libsrc 	src/activations.cpp
			src/batchnorm.cpp
			src/conversions.cpp
			src/convolution.cpp
			src/descriptors.cpp
			src/dropout.cpp
			src/fp16.cpp
			src/gemms.cpp
			src/im2row.cpp
			src/losses.cpp
			src/metrics.cpp
			src/optimizers.cpp
			src/pooling.cpp
			src/regularizers.cpp
			src/tensor_binary_op.cpp
			src/tensor_op.cpp
			src/tensor_reduction.cpp
			src/tensor_unary_op.cpp)

if(BUILD_SHARED_LIBS)
  set(LIBRARY_NAME "AvocadoReferenceBackend")
  add_library(${LIBRARY_NAME} SHARED ${libsrc})
  set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
  set(LIBRARY_NAME "AvocadoReferenceBackend_static")
  add_library(${LIBRARY_NAME} STATIC ${libsrc})
endif()

set_target_properties(${LIBRARY_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(${LIBRARY_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_MAIN_LIB}/include/Avocado/backend")
target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_CPU=0 USE_CUDA=0 USE_OPENCL=0)

install(TARGETS ${LIBRARY_NAME})
install(FILES "include/ReferenceBackend/reference_backend.h" DESTINATION include/ReferenceBackend)
install(FILES "${PATH_TO_MAIN_LIB}/include/Avocado/backend/backend_defs.h" DESTINATION include/ReferenceBackend)
