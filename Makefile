HL_PATH ?= ../FImage/cpp

sim: demo.c hlsolver.cpp
	g++ -framework OpenGL -framework GLUT -I${HL_PATH}/include -L${HL_PATH}/bin -lHalide $^ -o $@

test: sim
	HL_NUMTHREADS=2 DYLD_LIBRARY_PATH=${HL_PATH}/bin ./sim 180 0.05 0 0.0005 3 100

debug: sim
	DYLD_LIBRARY_PATH=${HL_PATH}/bin gdb ./sim
