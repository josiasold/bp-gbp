cmake_minimum_required(VERSION 3.16) 

project(bp_gbp
        VERSION 1.0
        LANGUAGES CXX)


# compiled library code
add_subdirectory(src)

# simulation executables
add_subdirectory(sim)

# "installation"
install(TARGETS sim
        DESTINATION ${PROJECT_SOURCE_DIR}/sim)

install(TARGETS sim_combs
        DESTINATION ${PROJECT_SOURCE_DIR}/sim)