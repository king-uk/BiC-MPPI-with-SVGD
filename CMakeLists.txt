cmake_minimum_required(VERSION 3.5)
project(mppi)

find_package(PythonLibs REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

include_directories(
    mppi/
    model/

    include/EigenRand/
    ${EIGEN3_INCLUDE_DIRS}

    # matplotlibcpp/
    ${PYTHON_INCLUDE_DIRS}

    # 현재 디렉토리의 하위 디렉토리에서 헤더 파일을 찾기 위해
    include/matplotlibcpp/

    inlcude/

)

 

    set(COMMON_SOURCES
        # src/wmrobot/bi_mppi.cpp
        # src/wmrobot/svgd_mppi.cpp
        # src/wmrobot/mppi.cpp
        # src/wmrobot/log_mppi.cpp
        # src/wmrobot/cluster_mppi.cpp
        # src/wmrobot/plotting.cpp
        # src/wmrobot/f1_tenth.cpp
        src/wmrobot/mppi_loop.cpp
        src/wmrobot/bi_loop.cpp
        src/wmrobot/cluster_mppi_loop.cpp
        src/wmrobot/svgd_loop.cpp
        # src/manipulator/bi_loop_7dof.cpp
        # src/manipulator/bi_loop_6dof.cpp
    )



foreach(SRC ${COMMON_SOURCES})
    get_filename_component(EXE_NAME ${SRC} NAME_WE)
    add_executable(${EXE_NAME} ${SRC})
    target_compile_options(${EXE_NAME} PRIVATE -O3)
    target_link_libraries(${EXE_NAME}     
        ${PYTHON_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
endforeach()