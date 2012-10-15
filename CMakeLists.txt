cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(MRSMap)

# In case the user does not setup CMAKE_BUILD_TYPE, assume it's RelWithDebInfo
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "build type default to RelWithDebInfo, set to Release to improve performance" FORCE)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  message(STATUS "building in release mode")
  add_definitions( -msse4.2 -fopenmp )
  add_definitions("-DBOOST_DISABLE_ASSERTS -DEIGEN_NO_DEBUG -DNDEBUG")
  add_definitions (-Wall -fno-strict-aliasing)# -fvisibility=hidden -fvisibility-inlines-hidden)
  add_definitions(-Wall -funroll-loops -momit-leaf-frame-pointer -fomit-frame-pointer -floop-block -ftree-loop-distribution -ftree-loop-linear -floop-interchange -floop-strip-mine -fgcse-lm -fgcse-sm -fsched-spec-load)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})


# PCL required
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OctreeLib required
find_package(OctreeLib REQUIRED)
include_directories(${OCTREELIB_INCLUDE_DIRS})

# GSL required
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
link_directories(${GSL_LIBRARY_DIRS})

# TBB required
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})

# OpenCV required
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# G2O required
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})

find_package( Boost 1.40 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )


# For building the CHOLMOD / CSPARSE solvers
find_package(Cholmod)
find_package(BLAS)
find_package(LAPACK)
find_package(CSparse)
include_directories(${CSPARSE_INCLUDE_DIR})


include_directories("/usr/include/openni/")

include_directories(${PROJECT_SOURCE_DIR}/include)



#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



# -- mrslib for map aggregation, registration, and visualization
set( SRC_MRSMAPLIB    
                src/map/multiresolution_csurfel_map.cpp
                src/registration/multiresolution_csurfel_registration.cpp
		src/utilities/utilities.cpp
		src/visualization/visualization_map.cpp
)

add_library( mrsmap SHARED ${SRC_MRSMAPLIB} )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( mrsmap octreelib ${PCL_LIBRARIES} ${OpenCV_LIBS} ${TBB_LIBRARIES} ${GSL_LIBRARIES})


# -- mrsslamlib for SLAM using mrs maps
add_library( mrsslam SHARED src/slam/slam.cpp src/visualization/visualization_slam.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( mrsslam mrsmap ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION} ${G2O_TYPES_SLAM3D})


# -- mrsslam_app for incremental slam using an openni camera
add_executable( mrsslam_app src/apps/mrsslam_app.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( mrsslam_app mrsslam )


# -- evaluate_visual_odometry for evaluation of visual odometry performance on RGBD dataset
add_executable( evaluate_visual_odometry src/apps/evaluate_visual_odometry.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( evaluate_visual_odometry mrsmap ${Boost_PROGRAM_OPTIONS_LIBRARY} )


# -- evaluate_slam for evaluation of SLAM performance on RGBD dataset
add_executable( evaluate_slam src/apps/evaluate_slam.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( evaluate_slam mrsslam )


# -- train_objectmap for mapping a multi-view mrs model of an object from RGBD dataset
add_executable( train_objectmap src/apps/train_objectmap.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( train_objectmap mrsslam ${Boost_PROGRAM_OPTIONS_LIBRARY} )


# -- evaluate_pose_tracking for evaluation of object tracking on RGBD dataset
add_executable( evaluate_pose_tracking src/apps/evaluate_pose_tracking.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( evaluate_pose_tracking mrsmap ${Boost_PROGRAM_OPTIONS_LIBRARY} )


# configure install target
set(CMAKE_INSTALL_PREFIX "/usr/local")

file(GLOB headers "${PROJECT_SOURCE_DIR}/include/mrsmap/map/*.h" "${PROJECT_SOURCE_DIR}/include/mrsmap/map/*.hpp")
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/mrsmap/map)

file(GLOB headers "${PROJECT_SOURCE_DIR}/include/mrsmap/registration/*.h" "${PROJECT_SOURCE_DIR}/include/mrsmap/registration/*.hpp")
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/mrsmap/registration)

file(GLOB headers "${PROJECT_SOURCE_DIR}/include/mrsmap/slam/*.h" "${PROJECT_SOURCE_DIR}/include/mrsmap/slam/*.hpp")
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/mrsmap/slam)

file(GLOB headers "${PROJECT_SOURCE_DIR}/include/mrsmap/utilities/*.h" "${PROJECT_SOURCE_DIR}/include/mrsmap/utilities/*.hpp")
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/mrsmap/utilities)

file(GLOB headers "${PROJECT_SOURCE_DIR}/include/mrsmap/visualization/*.h" "${PROJECT_SOURCE_DIR}/include/mrsmap/visualization/*.hpp")
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/mrsmap/visualization)

install(FILES "${LIBRARY_OUTPUT_PATH}/libmrsmap.so" "${LIBRARY_OUTPUT_PATH}/libmrsslam.so"
        DESTINATION lib)

install(FILES "${PROJECT_SOURCE_DIR}/cmake/Modules/FindMRSMap.cmake"
        DESTINATION /usr/share/cmake-2.8/Modules)


# configure uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)



