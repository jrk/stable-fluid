HL_PATH ?= ../FImage/cpp
GXX ?= g++

SIM_N ?= 190
SIM_DT ?= 0.05
SIM_DIFF ?= 0
SIM_VISC ?= 0.00005
SIM_FORCE ?= 3
SIM_SOURCE ?= 100

sim: demo.c hlsolver.cpp
	${GXX} -g -framework OpenGL -framework GLUT -I${HL_PATH}/include -L${HL_PATH}/bin -lHalide $^ -o $@

hlsolver: hlsolver.cpp
	${GXX} -g -DHL_STATIC_COMPILE=1 -I${HL_PATH}/include -L${HL_PATH}/bin -lHalide $^ -o $@

sim_static: demo.c hlsolver-static.cpp solver_dens_step.o solver_vel_step.o solver_dens_step.h solver_vel_step.h
	${GXX} -g -framework OpenGL -framework GLUT $^ -o $@

solver_dens_step.o solver_vel_step.o solver_dens_step.h solver_vel_step.h: hlsolver
	DYLD_LIBRARY_PATH=${HL_PATH}/bin ./$^ ${SIM_N} ${SIM_DT} ${SIM_DIFF} ${SIM_VISC} ${SIM_FORCE} ${SIM_SOURCE}

test: sim
	HL_NUMTHREADS=2 DYLD_LIBRARY_PATH=${HL_PATH}/bin ./$^ ${SIM_N} ${SIM_DT} ${SIM_DIFF} ${SIM_VISC} ${SIM_FORCE} ${SIM_SOURCE}

test_static: sim_static
	HL_NUMTHREADS=2 DYLD_LIBRARY_PATH=${HL_PATH}/bin ./$^ ${SIM_N} ${SIM_DT} ${SIM_DIFF} ${SIM_VISC} ${SIM_FORCE} ${SIM_SOURCE}

debug: sim
	DYLD_LIBRARY_PATH=${HL_PATH}/bin gdb ./sim
