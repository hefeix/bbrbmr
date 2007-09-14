COMPILE = $(CC) -O3 -c -Wno-deprecated -DTRAIN_ONLY -DPOLYTOMOUS -I../../.. -I$(TCLAPINCPATH) $(PROFILE)
#debug COMPILE = g++ -O3 -c -g -Wno-deprecated -DTRAIN_ONLY -I$(TCLAPINCPATH)

# C++ objects
CPPOBJ = DataFactory.o Matrix.o RunPoly.o PolyZO.o TestModel.o StrataSplit.o IndPriors.o
# QNOpt.o

#tclap is now all headers: %.o : $(TCLAPSRCPATH)/%.cpp
#	$(COMPILE) $< 

%.o : ../../%.cpp 
	$(COMPILE) $< 

#inherit from BBR
%.o : ../../../%.cpp 
	$(COMPILE) $< 

BMRtrain: $(CPPOBJ)
	$(CC) -O3 $(STATIC) -o $@ $(CPPOBJ) $() -lm $(PROFILE)

clean: 
	rm *.o 
	rm BMRtrain
