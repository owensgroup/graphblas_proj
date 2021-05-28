#ARCH=-gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60
ARCH=-gencode arch=compute_70,code=compute_70 -gencode arch=compute_70,code=sm_70
#ARCH=-gencode arch=compute_80,code=compute_80 -gencode arch=compute_80,code=sm_80
OPTIONS=-O3 -use_fast_math

all:  proj

proj: src/*
	nvcc -g $(ARCH) $(OPTIONS) -w -std=c++11 -o proj \
		src/proj.cu \
		--compiler-options "-fopenmp" \
		-lcusparse -lnvToolsExt \
		-Isrc/ 

clean:
	rm -f proj
