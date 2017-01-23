cmake_minimum_required(VERSION 2.8.3)

set(target_name face_verification_demo)

PROJECT(${target_name})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DBOOST_NO_CXX11_SCOPED_ENUMS -DCPU_ONLY")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DBOOST_NO_CXX11_SCOPED_ENUMS")

set(CAFFE_ROOT "/home/pegasw2/CV/ssd")

set(LINKER_LIBS "")

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${OpenCV_LIBS})

find_package(Boost COMPONENTS filesystem system REQUIRED)

include(${CAFFE_ROOT}/cmake/Modules/FindGlog.cmake)
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${GLOG_LIBRARIES})

include(${CAFFE_ROOT}/cmake/Modules/FindGFlags.cmake)
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${GFLAGS_LIBRARIES})

list(APPEND LINKER_LIBS ${CAFFE_ROOT}/build/lib/libcaffe.so)

include_directories(${CAFFE_ROOT}/include /usr/local/cuda/include)

###########
## Build ##
###########

# Tell cmake to link our target executable to dlib.
include(${CMAKE_CURRENT_SOURCE_DIR}/src/dlib/cmake)

# add all the cpp files we want to compile to this list.  This tells
# cmake that they are part of our target (which is the executable named face_roi_generator)
ADD_EXECUTABLE(${PROJECT_NAME}
	${CMAKE_CURRENT_SOURCE_DIR}/src/face_verification_demo.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/camera_cv_state.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/common.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/common.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/cv/face_verification.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/cv/face_verification.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/cv/caffe_face_verification.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/cv/caffe_face_verification.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/cv/caffe_face_detection.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/cv/caffe_face_detection.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/cv/caffe_face_landmark_detection.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/cv/caffe_face_landmark_detection.hpp
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}
	dlib
	${OpenCV_LIBS}
	${Boost_LIBRARIES}
	${LINKER_LIBS})
