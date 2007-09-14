CPPOBJ = DataFactory.o Matrix.o RunBatch.o TestModel.o

COMPILE = $(CC) -O3 -c -Wno-deprecated -DTEST_ONLY -DGROUPS -I$(TCLAPINCPATH)
# debug version COMPILE = g++ -O3 -c -g -Wno-deprecated -DTEST_ONLY -I$(TCLAPINCPATH)

%.o : ../../%.cpp
	$(COMPILE) $< 

BBRclassify: $(CPPOBJ)
	$(CC) -O3 $(STATIC) -o $@ $(CPPOBJ) $() -lm

clean: 
	rm *.o 
	rm BBRclassify
