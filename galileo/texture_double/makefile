LIBS=-lcuda -lglut -lm -L/usr/local/cuda/lib/ -lcudart
FLAGS=-std=c++11 -O3 -pg
#INCLUDE=-I/home/vicini/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/ -I/home/vicini/NVIDIA_GPU_Computing_SDK/shared/inc/ -I../../common/ -I/usr/local/cuda/include/
INCLUDE=
OBJS=-Wno-deprecated-gpu-targets

#NVCC=nvcc -Xptxas=-v,-abi=no -Xptxas=-fastimul
NVCC=nvcc -Xptxas=-v -Xptxas=-fastimul

ECHO=/bin/echo

default: MolDyn_NVE.x
 
%.x: %.cu
	$(NVCC) $(INCLUDE) $(OBJS) $(LIBS) $< -o $@ $(FLAGS)

clean:
	rm -f *.x *.o
copy:
	cp config.0 old.0
	cp config.final old.final
oldcopy:
	cp old.0 config.0
	cp old.final config.final
