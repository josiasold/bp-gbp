add_executable(sim sim.cpp)
add_executable(sim_w sim_w.cpp)
add_executable(sim_alpha sim_alpha.cpp)
add_executable(sim_KL sim_KL.cpp)
add_executable(sim_KL_repeatw sim_KL_repeatw.cpp)
add_executable(sim_KLll sim_KLll.cpp)
add_executable(sim_KL_nonCSS sim_KL_nonCSS.cpp)
add_executable(sim_combs sim_combs.cpp)
add_executable(sim_gf2 sim_gf2.cpp)


set(XTENSOR_USE_XSIMD 1)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/lib64")

set(xtensor_DIR "${CMAKE_SOURCE_DIR}/lib64/cmake/xtensor/")
set(xsimd_DIR "${CMAKE_SOURCE_DIR}/lib64/cmake/xsimd/")
set(xtl_DIR "${CMAKE_SOURCE_DIR}/share/cmake/xtl/")

message(STATUS "xtensor_DIR: ${xtensor_DIR}")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -g -O3 -march=native")

find_package(
    xtensor REQUIRED
)
target_compile_features(bp_gbp PUBLIC cxx_std_17)

target_link_libraries(sim PRIVATE bp_gbp m stdc++ stdc++fs xtensor)
target_link_libraries(sim_w PRIVATE bp_gbp m stdc++ stdc++fs xtensor)
target_link_libraries(sim_alpha PRIVATE bp_gbp m stdc++ stdc++fs xtensor)
target_link_libraries(sim_KL PRIVATE bp_gbp m stdc++ stdc++fs xtensor)
target_link_libraries(sim_KL_repeatw PRIVATE bp_gbp m stdc++ stdc++fs xtensor)
target_link_libraries(sim_KLll PRIVATE bp_gbp m stdc++ stdc++fs xtensor)
target_link_libraries(sim_KL_nonCSS PRIVATE bp_gbp m stdc++ stdc++fs xtensor)
target_link_libraries(sim_combs PRIVATE bp_gbp m stdc++ stdc++fs xtensor)
target_link_libraries(sim_gf2 PRIVATE bp_gbp m stdc++ stdc++fs xtensor)