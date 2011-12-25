COMPILE = $(CC) -O3 -c -Wno-deprecated -DTRAIN_ONLY -DGROUPS -I$(TCLAPINCPATH) -I$(TNTPATH) -I$(JAMAPATH)
#debug COMPILE = g++ -O3 -c -g -Wno-deprecated -DTRAIN_ONLY -DGROUPS -I$(TCLAPINCPATH) -I$(TNTPATH) -I$(JAMAPATH)

# C++ objects
CPPOBJ = DataFactory.o Matrix.o RunBatch.o StrataSplit.o TestModel.o FeatSelect.o ZO.o IndPriors.o HParSearch.o

# tclap is now all headers %.o : $(TCLAPSRCPATH)/%.cpp
#	$(COMPILE) $< 

%.o : ../../%.cpp 
	$(COMPILE) $< 

BBRtrain: $(CPPOBJ)
	$(CC) -O3 $(STATIC) -o $@ $(CPPOBJ) $() -lm

clean: 
	rm *.o 
	rm BBRtrain
