from ns_mh_phantom import *

print("Starting small test run...")

ns_out = run_ns_mh_phantom(
    n=100,
    p=3,
    n_live=20,
    data_seed=123
)

print("Run completed.")
print(type(ns_out))
