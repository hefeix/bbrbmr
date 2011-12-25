del *.obj
del *.pdb
set VCPATH=c:\Program Files\Microsoft Visual Studio 9.0\VC
set IDEPATH=c:\Program Files\Microsoft Visual Studio 9.0\Common7\IDE
set SDKLIB=c:\Program Files\Microsoft SDKs\Windows\v6.0A\Lib
rem set VCPATH=c:\Program Files\Microsoft Visual Studio .NET\Vc7
set INCLUDE=%VCPATH%\include;%VCPATH%\atlmfc\include;%VCPATH%\PlatformSDK\include\prerelease;%VCPATH%\PlatformSDK\include
set LIB=%VCPATH%\lib;%SDKLIB%
Path=%VCPATH%\bin;%IDEPATH%
cl.exe @bbrtrain.compile.rsp /nologo
link.exe @bbrtrain.link.rsp
cl.exe @bbrclassify.compile.rsp /nologo
link.exe @bbrclassify.link.rsp
