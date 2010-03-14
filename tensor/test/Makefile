BIN= test_example
OPT= -msse2 -O3 -fno-strict-aliasing -Wall


all: $(BIN)

clean:
	rm -f $(BIN) *.o *~

apex_tensor_cpu.o: ../apex_tensor.h ../apex_tensor_cpu.h ../apex_tensor_cpu.cpp ../apex_tensor_inline.cpp
	g++ $(OPT) -c -o apex_tensor_cpu.o  ../apex_tensor_cpu.cpp


test_example: tester_example.cpp apex_tensor_cpu.o 
	g++ $(OPT) -o test_example tester_example.cpp apex_tensor_cpu.o
