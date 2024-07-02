## Casadi codegen and compilation
Load function in whatever language
- `fn = casadi.Function.load('fn.casadi')`
Generate code
- `fn.generate('fn.c')`
Compile Casadi generated function
- `gcc -fPIC -shared -O3 -march=native fn.c -o fn.so`
Compile Casadi generated function with OpenMP
- `gcc -fPIC -shared -O3 -fopenmp -march=native fn.c -o fn.so`