# update with your own paths
export TCLAPINCPATH=$(HOME)/BBR/src/tclap-1.0.5/include
export TNTPATH=$(HOME)/BBR/src/tnt_3_0_4
export JAMAPATH=$(HOME)/BBR/src/jama124src

# export PROFILE=-pg
export PROFILE=

export CC = g++

all:
	cd bbrclassify && $(MAKE)
	mv bbrclassify/BBRclassify .
	cd bbrtrain && $(MAKE)
	mv bbrtrain/BBRtrain .
	cd bmrclassify && $(MAKE)
	mv bmrclassify/BMRclassify .
	cd bmrtrain && $(MAKE)
	mv bmrtrain/BMRtrain .

allstatic:
	cd bbrclassify && $(MAKE) STATIC=-static
	mv bbrclassify/BBRclassify ./static
	cd bbrtrain && $(MAKE) STATIC=-static
	mv bbrtrain/BBRtrain ./static
	cd bmrclassify && $(MAKE) STATIC=-static
	mv bmrclassify/BMRclassify ./static
	cd bmrtrain && $(MAKE) STATIC=-static
	mv bmrtrain/BMRtrain ./static

clean: 
	rm bbrclassify/*.o 
	rm bbrtrain/*.o
	rm bmrclassify/*.o
	rm bmrtrain/*.o
	rm BBRclassify
	rm BBRtrain
	rm BMRclassify
	rm BMRtrain
	rm ./static/*	