COMPILE = $(CC) -O0 -c -g -Wno-deprecated -DTRAIN_ONLY -DPOLYTOMOUS -I../../src -I$(TCLAPINCPATH) $(PROFILE)
#debug COMPILE = g++ -O3 -c -g -Wno-deprecated -DTRAIN_ONLY -I$(TCLAPINCPATH)

# C++ objects
CPPOBJ = DataFactory.o Matrix.o RunPoly.o PolyZO.o TestModel.o StrataSplit.o IndPriors.o
# QNOpt.o

#tclap is now all headers: %.o : $(TCLAPSRCPATH)/%.cpp
#	$(COMPILE) $< 

%.o : ../../src/BMR/%.cpp 
	$(COMPILE) $< 

#inherit from BBR
%.o : ../../src/%.cpp 
	$(COMPILE) $< 

BMRtrain: $(CPPOBJ)
	$(CC) -O0 -g $(STATIC) -o $@ $(CPPOBJ) $() -lm $(PROFILE)

clean: 
	rm *.o 
	rm BMRtrain
