cmake_minimum_required(VERSION 3.10)
project(ara_lightfield_demo LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GLFW
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)

# On many distros: sudo apt install libglfw3-dev

find_package(OpenGL REQUIRED)

# GLAD loader (generated from https://glad.dav1d.de/)
# OpenGL 4.5, Core profile, no extensions
include_directories(
    ${GLFW_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/external/glad/include
)

add_executable(ara_lightfield_demo
    src/main.cpp
    external/glad/src/glad.c
)

target_link_libraries(ara_lightfield_demo
    ${GLFW_LIBRARIES}
    ${OPENGL_gl_LIBRARY}
)

# For Linux you may need:
if(UNIX AND NOT APPLE)
    target_link_libraries(ara_lightfield_demo dl pthread)
endif()

# Copy shaders to build directory
file(GLOB SHADER_FILES "${CMAKE_SOURCE_DIR}/shaders/*")
foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    configure_file(${SHADER} ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME} COPYONLY)
endforeach()
