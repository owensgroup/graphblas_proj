ARCH=-gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60
OPTIONS=-O3 -use_fast_math

all: check-env proj

proj: src/*
	nvcc -g $(ARCH) $(OPTIONS) -w -std=c++11 -o proj \
		src/proj.cu \
		$(GRAPHBLAS_PATH)/ext/moderngpu/src/mgpucontext.cu \
		$(GRAPHBLAS_PATH)/ext/moderngpu/src/mgpuutil.cpp \
		-I$(GRAPHBLAS_PATH)/ext/moderngpu/include \
		-I$(GRAPHBLAS_PATH)/ext/cub/cub \
		-I$(GRAPHBLAS_PATH)/ \
		-Isrc/ \
		-I/usr/local/cuda/samples/common/inc/ \
		-lboost_program_options \
		-lcublas \
		-lcusparse \
		-lcurand

clean:
	rm -f proj
	
check-env:
ifndef GRAPHBLAS_PATH
	$(error `GRAPHBLAS_PATH` is undefined)
endif