# Rerun of r04b with a torch profiler on 3 steps (noise estimate + time breakdown)

Parent: r04b-d24w1408-lr5e3-unet-nest. Identical training; profiles steps 30-32 on rank 0 and prints the top CUDA kernels. Gives the run-to-run noise level and where the 0.74 s/step goes (to find throughput wins -> bigger model).
