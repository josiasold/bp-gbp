add_executable(sim sim.cpp)
add_executable(sim_KL sim_KL.cpp)


set(xtensor_DIR "${CMAKE_SOURCE_DIR}/lib64/cmake/xtensor/")
set(xsimd_DIR "${CMAKE_SOURCE_DIR}/lib64/cmake/xsimd/")
set(xtl_DIR "${CMAKE_SOURCE_DIR}/share/cmake/xtl/")

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/lib64")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native")

find_package(OpenMP)
find_package(xtensor REQUIRED)
message(STATUS "xtensor_DIR: ${xtensor_DIR}")

set(XTENSOR_USE_XSIMD 1)

target_compile_features(bp_gbp PUBLIC cxx_std_17)

if(OpenMP_CXX_FOUND)
    target_link_libraries(sim_KL PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(sim PRIVATE bp_gbp m stdc++ stdc++fs xtensor)
target_link_libraries(sim_KL PRIVATE bp_gbp m stdc++ stdc++fs xtensor)