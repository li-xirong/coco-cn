call "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\vcvarsall.bat" x86_amd64

nmake -f Makefile.win64 clean
nmake -f Makefile.win64 all
@pause

