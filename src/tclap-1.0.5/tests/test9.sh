#!/bin/sh

# success
../examples/test2 -i 10 -s hello goodbye -- -hv one two > tmp.out 2>&1

if cmp -s tmp.out test9.out; then
	exit 0
else 
	exit 1
fi

