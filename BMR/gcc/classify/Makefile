COMPILE = $(CC) -O3 -c -Wno-deprecated -DTEST_ONLY -DPOLYTOMOUS -I../../.. -I$(TCLAPINCPATH)
# debug version COMPILE = g++ -O3 -c -g -Wno-deprecated -DTEST_ONLY -I$(TCLAPINCPATH)

CPPOBJ = DataFactory.o RunPoly.o TestModel.o

#tclap is now all headers: %.o : $(TCLAPSRCPATH)/%.cpp
#	$(COMPILE) $< 

%.o : ../../%.cpp
	$(COMPILE) $< 

BMRclassify: $(CPPOBJ)
	$(CC) -O3 $(STATIC) -o $@ $(CPPOBJ) $() -lm

clean: 
	rm *.o 
	rm BMRclassify
