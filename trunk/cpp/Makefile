BIN= srbm_minist
OPT= -msse2 -O3 -fno-strict-aliasing -Wall
TENSOR_DIR=../tensor
SRBM_DIR=../srbm 
all: $(BIN)

clean:
	rm -f $(BIN) *.o *~ *.exe

apex_srbm.o:  $(SRBM_DIR)/apex_srbm.cpp 
	g++ $(OPT) -c -o apex_srbm.o  $(SRBM_DIR)/apex_srbm.cpp

apex_tensor_cpu.o: $(TENSOR_DIR)/apex_tensor.h $(TENSOR_DIR)/apex_tensor_cpu.h $(TENSOR_DIR)/apex_tensor_cpu.cpp $(TENSOR_DIR)/apex_tensor_inline.cpp
	g++ $(OPT) -c -o apex_tensor_cpu.o  $(TENSOR_DIR)/apex_tensor_cpu.cpp

srbm_minist: srbm_minist.cpp apex_tensor_cpu.o  apex_srbm.o
	g++ $(OPT) -o srbm_minist srbm_minist.cpp apex_tensor_cpu.o apex_srbm.o 
