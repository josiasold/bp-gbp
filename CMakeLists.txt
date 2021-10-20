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

install(TARGETS sim_w
        DESTINATION ${PROJECT_SOURCE_DIR}/sim)

install(TARGETS sim_alpha
        DESTINATION ${PROJECT_SOURCE_DIR}/sim)

install(TARGETS sim_KL
        DESTINATION ${PROJECT_SOURCE_DIR}/sim)

install(TARGETS sim_KL_repeatw
        DESTINATION ${PROJECT_SOURCE_DIR}/sim)

install(TARGETS sim_KLll
        DESTINATION ${PROJECT_SOURCE_DIR}/sim)

install(TARGETS sim_KL_nonCSS
        DESTINATION ${PROJECT_SOURCE_DIR}/sim)

install(TARGETS sim_combs
        DESTINATION ${PROJECT_SOURCE_DIR}/sim)

install(TARGETS sim_gf2
        DESTINATION ${PROJECT_SOURCE_DIR}/sim)