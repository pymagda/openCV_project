cmake_minimum_required(VERSION 3.3)
project(openCV_project)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "build")
set(QT5_PATH $ENV{HOME}/Qt/5.5/clang_64/ CACHE PATH "Path to Qt5")
set(QT5_MODULE_PATH ${QT5_PATH}/lib/cmake)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${QT5_MODULE_PATH})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

#configure_file(krasnal.png ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/krasnal.png COPYONLY)

find_package(OpenCV REQUIRED)
find_package(Qt5Widgets)
find_package(Qt5Core)
find_package(Qt5Gui)

include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

add_library(mainwindow Frontend/mainwindow.cpp)
target_link_libraries (mainwindow Qt5::Widgets)

set(SOURCE_FILES Backend/Source/CmPreProcess.cpp Backend/Source/CmSaliency.cpp Backend/Source/CmCut.cpp Backend/main.cpp)
add_executable(openCV_project ${SOURCE_FILES})
target_link_libraries(openCV_project ${OpenCV_LIBS} mainwindow)