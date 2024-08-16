## Casadi codegen and compilation
Load function in whatever language
- `fn = casadi.Function.load('fn.casadi')`
Generate code
- `fn.generate('fn.c')`
Compile Casadi generated function
- `gcc -fPIC -shared -O3 -march=native fn.c -o fn.so`
Compile Casadi generated function with OpenMP
- `gcc -fPIC -shared -O3 -fopenmp -march=native fn.c -o fn.so`

## C++ compilation for Casadi evaluation

Requires Eigen and system-installed Casadi
- Casadi: https://github.com/casadi/casadi/wiki/InstallationLinux
- Eigen: https://eigen.tuxfamily.org/index.php?title=Main_Page#Download

`g++ -O3 -std=c++11 -fPIC -fopenmp -o [EXEC_OUT] [FILE_IN] -I/usr/local/include/casadi -L/usr/local/lib/libcasadi.so -lcasadi`
`nvcc -O3 -std=c++11 -Xcompiler -fPIC -Xcompiler -fopenmp -o test evaluate_serial_cpu.cpp -I/usr/local/include/casadi -L/usr/local/lib/libcasadi.so -lcasadi`