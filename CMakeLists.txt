cmake_minimum_required(VERSION 2.6)

project(sep C)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules/")
include(GNUInstallDirs)

set(SOURCES
   ${CMAKE_SOURCE_DIR}/src/analyse.c
   ${CMAKE_SOURCE_DIR}/src/convolve.c
   ${CMAKE_SOURCE_DIR}/src/deblend.c
   ${CMAKE_SOURCE_DIR}/src/extract.c
   ${CMAKE_SOURCE_DIR}/src/lutz.c
   ${CMAKE_SOURCE_DIR}/src/aperture.c
   ${CMAKE_SOURCE_DIR}/src/background.c
   ${CMAKE_SOURCE_DIR}/src/psf.c
   ${CMAKE_SOURCE_DIR}/src/util.c
   )

include_directories(${CMAKE_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/src ${CFITSIO_INCLUDE_DIR})

link_directories(${CMAKE_LIBRARY_PATH} ${CMAKE_SOURCE_DIR}/src)

add_library(sep SHARED ${SOURCES})
set_target_properties(sep PROPERTIES OUTPUT_NAME sep)
set_target_properties(sep PROPERTIES VERSION 0.6.0 SOVERSION 0)
set_target_properties(sep PROPERTIES C_VISIBILITY_PRESET hidden)

option(SEP_USE_OPENMP "Enable OpenMP acceleration" ON)
if (SEP_USE_OPENMP)
   find_package(OpenMP QUIET)
   if (OpenMP_C_FOUND)
      if (OpenMP_C_FLAGS)
         set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      endif()
      if (OpenMP_C_LIBRARIES)
         target_link_libraries(sep ${OpenMP_C_LIBRARIES})
      endif()
   endif()
endif()

if (MSVC)
   add_definitions(-D_USE_MATH_DEFINES)
else ()
   add_compile_options(-Wcast-qual)
   target_link_libraries(sep m)
endif()

install(TARGETS sep LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${CMAKE_SOURCE_DIR}/src/sep.h DESTINATION include)
