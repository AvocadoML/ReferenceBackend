cmake_minimum_required(VERSION 3.18)
project(ReferenceBackend VERSION 1.0.0 DESCRIPTION "Reference backend implementation" LANGUAGES CXX)

option(BUILD_TESTING "Build unit tests?" OFF)
set(PATH_TO_MAIN_LIB "Path to the main Avocado project" "${PROJECT_SOURCE_DIR}/../Avocado")

add_library(ReferenceBackend STATIC)
set_target_properties(ReferenceBackend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(ReferenceBackend PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(ReferenceBackend PUBLIC "${PATH_TO_MAIN_LIB}/include")

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
	add_executable(ReferenceTests)
	set_target_properties(ReferenceTests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
	target_link_libraries(ReferenceTests PRIVATE ReferenceBackend)
	add_subdirectory(test)
endif()

add_subdirectory(src)