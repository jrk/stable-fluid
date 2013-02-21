HL_PATH ?= ../FImage/cpp
GXX ?= g++

sim: demo.c hlsolver.cpp
	${GXX} -g -framework OpenGL -framework GLUT -I${HL_PATH}/include -L${HL_PATH}/bin -lHalide $^ -o $@

test: sim
	HL_NUMTHREADS=2 DYLD_LIBRARY_PATH=${HL_PATH}/bin ./sim 190 0.05 0 0.0005 3 100

debug: sim
	DYLD_LIBRARY_PATH=${HL_PATH}/bin gdb ./sim
