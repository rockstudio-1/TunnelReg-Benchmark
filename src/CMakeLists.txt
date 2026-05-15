# =============================================================================
#  TunnelReg-Benchmark - classical baselines (FPFH/CSHOT/ICP)
#
#  Copyright (c) 2026 Shibin Tang, Dalian University of Technology
#  Licensed under the MIT License.
# =============================================================================
cmake_minimum_required(VERSION 3.10)
project(tunnelreg_eval LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(PCL 1.10 REQUIRED
    COMPONENTS
        common
        io
        features
        registration
        visualization
        filters
        search
)

find_package(OpenMP)

include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------
add_executable(eval eval.cpp)
target_link_libraries(eval PRIVATE ${PCL_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(eval PRIVATE OpenMP::OpenMP_CXX)
endif()

# Optional: enable -O3 / /O2 (CMake's Release already does this, but keep
# explicit warnings for cleaner output).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(eval PRIVATE -Wall -Wno-deprecated-declarations)
elseif(MSVC)
    target_compile_options(eval PRIVATE /W3 /wd4996)
endif()
