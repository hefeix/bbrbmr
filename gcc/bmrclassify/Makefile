COMPILE = $(CC) -O0 -g -c -Wno-deprecated -DTEST_ONLY -DPOLYTOMOUS -I../../src -I$(TCLAPINCPATH)
# debug version COMPILE = g++ -O3 -c -g -Wno-deprecated -DTEST_ONLY -I$(TCLAPINCPATH)

CPPOBJ = DataFactory.o RunPoly.o TestModel.o

#tclap is now all headers: %.o : $(TCLAPSRCPATH)/%.cpp
#	$(COMPILE) $< 

%.o : ../../src/BMR/%.cpp
	$(COMPILE) $< 

BMRclassify: $(CPPOBJ)
	$(CC) -O3 $(STATIC) -o $@ $(CPPOBJ) $() -lm

clean: 
	rm *.o 
	rm BMRclassify
