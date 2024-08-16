# import sys
# import time
# import casadi
# import numpy as np
# import subprocess

# t_parallel = subprocess.run([
#     './evaluate_parallel_cpu', 'fn_1e5', 'fn_1e5.so', '1000'],
#     capture_output=True,
#     text=True)
# t_serial = subprocess.run([
#     './evaluate_serial_cpu', 'fn_1e5', 'fn_1e5.so', '1000'],
#     cwd="/home/sehwan/Research/cusadi/src/benchmark_functions/",
#     capture_output=True,
#     text=True)
# print(float(t_parallel.stdout.strip()))
# print(float(t_serial.stdout.strip()))


# # N_ENVS = 10000
# # # fn = casadi.external('fn_1e5', 'fn_1e5.so')
# # fn = casadi.Function.load('fn_1e5.casadi')
# # fn = fn.expand()
# # f_mapped_name = f"{fn.name()}_mapped_{N_ENVS}"
# # c_filepath = f"{f_mapped_name}.c"
# # so_filepath = f"{f_mapped_name}.so"
# # f_serial_name = f"{fn.name()}_serial_{N_ENVS}"
# # f_serial = fn.map(N_ENVS)
# # f_serial.generate(f"{f_serial_name}.c")

# # f_parallel_name = f"{fn.name()}_parallel_{N_ENVS}"
# # f_parallel = fn.map(N_ENVS, "thread", 16)
# # f_parallel.generate(f"{f_parallel_name}.c")

# # # os.system(f"gcc -fPIC -shared -O3 -fopenmp -march=native {c_filepath} -o {so_filepath}")
# # print("Serial function name: ", f_serial.name())
# # print("Parallel function name: ", f_parallel.name())

# # fn_serial = casadi.external('map1000_fn_1e5', "fn_1e5_serial_1000.so")
# # # fn_parallel = casadi.external('helper', 'fn_1e5_parallel_1000.so')
# # inputs_CPU = [casadi.DM(np.random.rand(fn.nnz_in(i), N_ENVS)) for i in range(fn.n_in())]
# # inputs_single = [inputs_CPU[j][:, 0] for j in range(len(inputs_CPU))]

# # # start = time.process_time()
# # start = time.time()
# # out = fn_serial(inputs_CPU[0], inputs_CPU[1], inputs_CPU[2], inputs_CPU[3])
# # end = time.time()
# # # end = time.process_time()
# # print("Time taken for evaluation:", end - start)

# # start = time.process_time()
# # out = f_serial(inputs_CPU[0], inputs_CPU[1], inputs_CPU[2], inputs_CPU[3])
# # end = time.process_time()
# # print("Time taken for evaluation:", end - start)