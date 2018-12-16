CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC

SEARCHLIB_NAME = libsearch


all: libsearch 


libsearch: search.o
	$(CXX) -shared -dynamiclib search.o -o $(SEARCHLIB_NAME).so

search.o: search.cpp search.h
	$(CXX) $(CFLAGS) -c search.cpp
clean:
	rm -f *~ *.o *.so
