cmake_minimum_required(VERSION 2.8)
project( cgsr )

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

find_package( OpenCV REQUIRED core imgproc calib3d optflow highgui)

file(GLOB SOURCE_FILES
    "src/*.h"
    "src/*.cpp"
)
        
add_executable( cgsr ${SOURCE_FILES} )
set_property(TARGET cgsr PROPERTY CXX_STANDARD 14)
set_property(TARGET cgsr PROPERTY CXX_STANDARD_REQUIRED ON)
target_include_directories( cgsr PUBLIC ${OpenCV_INCLUDE_DIRS})

target_compile_options(cgsr PUBLIC -Wno-deprecated-declarations)

target_compile_options(cgsr PUBLIC -mavx)
message(STATUS "Enabling AVX instructions")
target_compile_options(cgsr PUBLIC -msse4)
message(STATUS "Enabling SSE4 instructions")
target_compile_options(cgsr PUBLIC -msse2)
message(STATUS "Enabling SSE2 instructions")

target_compile_options(cgsr PUBLIC -march=native)    

target_compile_definitions(cgsr PUBLIC EIGEN_DONT_ALIGN_STATICALLY)

target_include_directories(cgsr PUBLIC /usr/include/eigen3)
target_link_libraries( cgsr ${OpenCV_LIBS} pthread ceres gomp glog cxsparse blas lapack cholmod glfw GL)



