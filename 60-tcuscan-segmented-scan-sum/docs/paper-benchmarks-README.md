# Benchmarks and plots for ALENEX 26 submission

## 1) Requirements - Prepare all baselines
1. First, we need to build the package with `make build`.
2. We need to build EIGEN for spmv benchmark. To do this navigate to the folder `scripts/baselines/eigen-spmv-cpu` and run `make compile`
3. We need to install MKL. Currently we use the pre-build shared objects shipped with conda. To install MKL, navigate to `scripts/baselines/mkl` and do `conda env create -f environment.yml`. It requires a working conda distribution.
4. For the segmented scan CPU comparison, we need to build the baseline code under `scripts/baselines/segscan_cpu`. Inside that folder, run `make compile`.

## 2) Executing the benchmarks

### fig_6_2_comparison_CPU_VCU_MMU
- To prepare the csv files: `make paper_fig_3`. 
- Copy the `.csv` files inside `nbs/paper-plots/alenex26/fig_6_2_comparison_CPU_VCU_MMU/`.
- Run the notebook to make the plot.

### fig_6_3_density_comparison
- To prepare csv files: `make paper_fig_4b`. 
- Copy the `.csv` files inside `nbs/paper-plots/alenex26/fig_6_3_density_comparison/`.
- Run the notebook to make the plot.

### fig_6_4_breakdown_seg_ops
- To prepare csv files: `make paper_fig_6`.
- Copy the `.csv` files inside `nbs/paper-plots/alenex26/fig_6_4_breakdown_seg_ops/` .
- Run the notebook to make the plot.

### fig_6_5_bandwidth_throughput
- To prepare csv files: `make paper_fig_7`.
- Copy the `.csv` files inside `nbs/paper-plots/alenex26/fig_6_5_bandwidth_throughput/` .
- Run the notebook to make the plot.

### fig_6_6_spmv
This is deprecated, see `fig_6_6_spmv_new`

### fig_6_6_spmv_new
- To prepare csv files: `cd nbs/paper-plots/alenex26/fig_6_6_spmv_new` and run `./run_all_benchmarks.sh`
- Run the notebook to make the plot.
