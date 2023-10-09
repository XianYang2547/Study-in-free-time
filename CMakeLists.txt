cmake_minimum_required(VERSION 3.26)
project(C_Lib)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

set(CMAKE_CXX_STANDARD 17)
add_executable(C_Lib main.cpp
        test.cpp)

set(CMAKE_PREFIX_PATH E:/libtorch)
set(OpenCV_DIR E:/opencv/build/x64/vc16/lib)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(C_Lib ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET C_Lib PROPERTY CXX_STANDARD 11)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET C_Lib
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:C_Lib>)

    file(GLOB OPENCV_DLLS "E:/opencv/build/x64/vc16/bin/*.dll")
    add_custom_command(TARGET C_Lib
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS}
            $<TARGET_FILE_DIR:C_Lib>)
endif (MSVC)