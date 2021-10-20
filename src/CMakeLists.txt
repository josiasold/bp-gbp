file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${bp_gbp_SOURCE_DIR}/include/bp_gbp/*.hpp")

message(STATUS "bp_gbp_SOURCE_DIR: ${bp_gbp_SOURCE_DIR}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "HEADER_LIST: ${HEADER_LIST}")

add_library(bp_gbp decoder_bp2.cpp decoder_bp_kuolai.cpp decoder_bp_klm.cpp decoder_bp.cpp constructor_gbp.cpp decoder_gbp.cpp error_channels.cpp io_tools.cpp la_tools.cpp timing.cpp ${HEADER_LIST})

target_include_directories(bp_gbp PUBLIC ../include)
target_include_directories(bp_gbp PUBLIC ../lib)
target_include_directories(bp_gbp PUBLIC ../lib64)


# find_library(
#     GMP
#     NAMES gmp libgmp
# )

find_library(
    NTL
    NAMES ntl libntl
    HINTS "${CMAKE_SOURCE_DIR}/lib"
)

if (NOT NTL)
    message(SEND_ERROR "Did not find lib ntl")
endif ()

find_library(
    LEMON
    NAMES lemon liblemon emon libemon
    HINTS "${CMAKE_SOURCE_DIR}/lib"
)

if (NOT LEMON)
    message(SEND_ERROR "Did not find lib lemon")
endif ()

#message(STATUS "GMP: ${GMP_INCLUDE_DIRS}")


set(XTENSOR_USE_XSIMD 1)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/lib64")

set(xtensor_DIR "${CMAKE_SOURCE_DIR}/lib64/cmake/xtensor/")
set(xsimd_DIR "${CMAKE_SOURCE_DIR}/lib64/cmake/xsimd/")
set(xtl_DIR "${CMAKE_SOURCE_DIR}/share/cmake/xtl/")

message(STATUS "xtensor_DIR: ${xtensor_DIR}")

find_package(
    xtensor REQUIRED
)

target_compile_features(bp_gbp PUBLIC cxx_std_17)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native")
# SET(PLATFORM_SPECIFIC_LIBS "-lpthread")

target_link_libraries(bp_gbp PUBLIC ${LEMON} stdc++ stdc++fs ${NTL} gmp m xtensor)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files" 
    FILES ${HEADER_LIST}
)