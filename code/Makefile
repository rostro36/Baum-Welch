#COMPILER
CC = gcc
#COMPILER FLAGS
CFLAGS = -O2 -Wall
#LIBRARIES TO LINK
LIBS = -lm
#FLAGS FOR VECTORIZATION
VECFLAGS = -mfma
#ROOT OF MKL FOR BLAS
MKLROOT = /opt/intel/mkl
#ADDITIONAL FLAGS FOR BLAS
BLASFLAGS = -DMKL_ILP64 -m64 -I$(MKLROOT)/include
#ADDITIONAL LINKING FOR BLAS
BLASLIBS = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -ldl
#DEPENDENCIES
DEPS = io.h tested.h util.h
#OBJECTIVES
OBJ = io.o bw-tested.o util.o

.PHONY: all clean clean_all

all: stb cop reo vec bla

#DEFAULT COMPILATION
%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $< 

#LINKING ALL TOGETHER
stb: bw-stb.o $(OBJ) 
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

#LINKING ALL TOGETHER
cop: bw-cop.o $(OBJ) 
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

#LINKING ALL TOGETHER
reo: bw-reo.o $(OBJ) 
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

#COMPILATION OF VEC (NEEDS ADDITIONAL FLAG)
bw-vec.o: bw-vec.c $(DEPS)
	$(CC) $(CFLAGS) $(VECFLAGS) -c -o $@ $< 

#LINKING ALL TOGETHER
vec: bw-vec.o $(OBJ) 
	$(CC) $(CFLAGS) $(VECFLAGS) -o $@ $^ $(LIBS)

#COMPILATION OF BLAS (NEEDS ADDITIONAL FLAG)
bw-bla.o: bw-bla.c $(DEPS)
	$(CC) $(CFLAGS) $(BLASFLAGS) -c -o $@ $< 
	
#LINKING ALL TOGETHER
bla: bw-bla.o $(OBJ) 
	$(CC) $(CFLAGS) $(BLASFLAGS) -o $@ $^ $(BLASLIBS) $(LIBS)

#FOR OTHER VERSIONS (e.g. cachegrind)
#LINKING ALL TOGETHER
stb%: bw-stb%.o $(OBJ) 
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

#LINKING ALL TOGETHER
cop%: bw-cop%.o $(OBJ) 
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

#LINKING ALL TOGETHER
reo%: bw-reo%.o $(OBJ) 
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

#COMPILATION OF VEC (NEEDS ADDITIONAL FLAG)
bw-vec%.o: bw-vec%.c $(DEPS)
	$(CC) $(CFLAGS) $(VECFLAGS) -c -o $@ $< 

#LINKING ALL TOGETHER
vec%: bw-vec%.o $(OBJ) 
	$(CC) $(CFLAGS) $(VECFLAGS) -o $@ $^ $(LIBS)

#COMPILATION OF BLAS (NEEDS ADDITIONAL FLAG)
bw-bla%.o: bw-bla%.c $(DEPS)
	$(CC) $(CFLAGS) $(BLASFLAGS) -c -o $@ $< 
	
#LINKING ALL TOGETHER
bla%: bw-bla%.o $(OBJ) 
	$(CC) $(CFLAGS) $(BLASFLAGS) -o $@ $^ $(BLASLIBS) $(LIBS)


#CLEANING UP
clean:
	rm -f bw-stb.o
	rm -f stb
	rm -f bw-cop.o
	rm -f cop
	rm -f bw-reo.o
	rm -f reo
	rm -f bw-vec.o
	rm -f vec
	rm -f bw-bla.o
	rm -f bla
	rm -f bw-stb*.o
	rm -f stb*
	rm -f bw-cop*.o
	rm -f cop*
	rm -f bw-reo*.o
	rm -f reo*
	rm -f bw-vec*.o
	rm -f vec*
	rm -f bw-bla*.o
	rm -f bla*
	
clean_all: clean
	rm -f bw-tested.o
	rm -f io.o
	rm -f util.o
