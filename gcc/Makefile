# update with your own paths
export TCLAPINCPATH=../../tclap-1.0.5/include
export TNTPATH=../../tnt_3_0_4
export JAMAPATH=../../jama124src

# export PROFILE=-pg
export PROFILE=

export CC = g++

BBR:
	cd classify && $(MAKE)
	mv classify/BBRclassify .
	cd train && $(MAKE)
	mv train/BBRtrain .

BBRstatic:
	cd classify && $(MAKE) STATIC=-static
	mv classify/BBRclassify ./static
	cd train && $(MAKE) STATIC=-static
	mv train/BBRtrain ./static

clean: 
	rm classify/*.o 
	rm train/*.o
	rm BBRclassify
	rm BBRtrain
