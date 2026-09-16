# Core logic: model loading, router probing, vLLM serving, workload
# construction, forced imbalance and trace analysis.
#
# The modules in this package are imported lazily by the pipelines, after
# moe_reliability.environment.configure_environment has imported
# torch_npu and enabled transfer_to_npu, so that device-generic PyTorch
# calls in this code run on Ascend NPUs.

