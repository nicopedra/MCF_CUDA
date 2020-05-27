LIBS=-lcuda -lglut -lm -L/usr/local/cuda/lib/ -lcudart
FLAGS=-std=c++11 -O3 -pg
INCLUDE=-I/home/vicini/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/ -I/home/vicini/NVIDIA_GPU_Computing_SDK/shared/inc/ -I../../common/ -I/usr/local/cuda/include/
OBJS=-Wno-deprecated-gpu-targets

NVCC=nvcc -arch=sm_20

ECHO=/bin/echo

default: MolDyn_NVE.x

 
%.x: %.cu
	$(NVCC) $(INCLUDE) $(OBJS) $(LIBS) $< -o $@ $(FLAGS)

clean:
	rm -f *.x *.o
copy:
	cp config.0 old.0
	cp config.final old.final

