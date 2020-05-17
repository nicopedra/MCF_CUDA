FLAGS=-std=c++11 -O3 -pg #-Q 

compila: MolDyn_NVE.cpp MolDyn_NVE.h
	g++ MolDyn_NVE.cpp -o MolDyn_NVE.x  $(FLAGS)

clean:
	rm -f *.x *.o
copy:
	cp config.0 old.0
	cp config.final old.final

