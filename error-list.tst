OSError: [WinError 126] The specified module could not be found. Error loading "D:\OBJECT DETECTION\.venv\lib\site-packages\torch\lib\fbgemm.dll" or 
one of its dependencies.

solve : But the issue solved by simply downloading libomp140.x86_64.dll and place it in 'C:\Windows\System32' and it finally works after long pain of trying this and that!