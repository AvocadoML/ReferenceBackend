cmake_minimum_required(VERSION 3.18)
project(ReferenceBackend VERSION 1.0.0 DESCRIPTION "Reference backend implementation" LANGUAGES CXX)

option(BUILD_TESTING "Build unit tests?" OFF)
option(BUILD_SHARED_LIBS "Build shared library?" OFF)
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Choose debug postfix")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type")
set(PATH_TO_COMMON_UTILS "${PROJECT_SOURCE_DIR}/../CommonBackendUtils" CACHE STRING "Path to the common backend utils project")

set(commonsrc 	${PATH_TO_COMMON_UTILS}/src/descriptors/ContextDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/ConvolutionDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/DropoutDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/MemoryDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/OptimizerDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/PoolingDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/TensorDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/backend_utils.cpp)

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
			src/tensor_unary_op.cpp
			src/winograd_transforms.cpp)

if(BUILD_SHARED_LIBS)
  set(LIBRARY_NAME "AvocadoReferenceBackend")
  add_library(${LIBRARY_NAME} SHARED ${libsrc} ${commonsrc})
  set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
  set(LIBRARY_NAME "AvocadoReferenceBackend_static")
  add_library(${LIBRARY_NAME} STATIC ${libsrc} ${commonsrc})
endif()

set_target_properties(${LIBRARY_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(${LIBRARY_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_COMMON_UTILS}/include")

install(TARGETS ${LIBRARY_NAME})
install(FILES "include/Avocado/reference_backend.h" DESTINATION include/Avocado)
