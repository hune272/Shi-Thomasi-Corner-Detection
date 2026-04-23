cmake_minimum_required(VERSION 3.20)
project(ImageProcessingProject)

set(CMAKE_CXX_STANDARD 17)

set(OPENCV_DIR ${CMAKE_SOURCE_DIR}/OpenCV)
set(OPENCV_INCLUDE ${OPENCV_DIR}/include)
set(OPENCV_LIB_DIR ${OPENCV_DIR}/lib)

include_directories(${OPENCV_INCLUDE})
link_directories(${OPENCV_LIB_DIR})

add_executable(${PROJECT_NAME}
    stdafx.cpp
    common.cpp
    ShiTomasi.cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    _CONSOLE
    _CRT_SECURE_NO_WARNINGS
)

target_link_libraries(${PROJECT_NAME}
    $<$<CONFIG:Debug>:opencv_world490d>
    $<$<CONFIG:Release>:opencv_world490>
)

# Copy DLLs to output directory after build
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${OPENCV_DIR}/dll/$<IF:$<CONFIG:Debug>,Debug,Release>
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
)
