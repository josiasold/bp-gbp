add_executable(sim sim.cpp)

add_executable(sim_combs sim_combs.cpp)

target_compile_features(bp_gbp PUBLIC cxx_std_17)

target_link_libraries(sim PRIVATE bp_gbp m stdc++ stdc++fs)
target_link_libraries(sim_combs PRIVATE bp_gbp m stdc++ stdc++fs)