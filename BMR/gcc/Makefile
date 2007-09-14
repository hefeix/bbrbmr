# update with your own paths
export TCLAPINCPATH=$(HOME)/source/tclap-1.0.5/include
#export PROFILE=-pg -g
export PROFILE=

export CC = g++

BMR:
	cd train && $(MAKE)
	mv train/BMRtrain .
	cd classify && $(MAKE)
	mv classify/BMRclassify .

BMRstatic:
	cd train && $(MAKE) STATIC=-static
	mv train/BMRtrain ./static
	cd classify && $(MAKE) STATIC=-static
	mv classify/BMRclassify ./static

clean: 
	rm train/*.o
	rm BMRtrain
	rm classify/*.o 
	rm BMRclassify
