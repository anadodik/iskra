cmake_minimum_required(VERSION 3.18...3.27)

project(iskra LANGUAGES C CXX)  # CUDA

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

if(SKBUILD)
    message(STATUS "Using scikit-build: ${PYTHON_SITE_PACKAGES_DIR}, ${CONDA_ENV_LIB_DIR}, ${NB_DIR}")
endif()

nanobind_build_library(nanobind SHARED)

nanobind_add_module(io_ext STABLE_ABI csrc/io.cpp)
target_include_directories(io_ext PRIVATE csrc/ext)
install(TARGETS io_ext LIBRARY DESTINATION iskra/io)
nanobind_add_stub(io_ext_stub MODULE io_ext OUTPUT io_ext.pyi PYTHON_PATH $<TARGET_FILE_DIR:io_ext> DEPENDS io_ext)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/io_ext.pyi" DESTINATION iskra/io)