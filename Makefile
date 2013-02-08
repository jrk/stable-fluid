sim: solver.c demo.c
	g++ -framework OpenGL -framework GLUT $^ -o $@

test: sim
	./sim 150 0.05 0 0.001 3 100
