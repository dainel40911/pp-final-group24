cmake_minimum_required(VERSION 3.16)
project(pp_final LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -mavx2")

include_directories(include)

add_executable(app
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(app PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(app
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/bin"
)