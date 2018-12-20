CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC -ftree-vectorize -msse3 -ffast-math -fassociative-math -I/usr/local/Cellar/boost/1.58.0/include/ -static

SEARCHLIB_NAME = libsearch


all: libsearch


libsearch: search.o
	$(CXX) -shared -dynamiclib search.o -o $(SEARCHLIB_NAME).so
	rm -f *.o

search.o: search.cpp search.h
	$(CXX) $(CFLAGS) -c search.cpp

clean:
	rm -f *~ *.o *.so
