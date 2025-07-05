#include <torch/extension.h>

// A simple example: add 1 to every element
torch::Tensor add_one(const torch::Tensor& x) {
    return x + 1;
}

// add random Gaussian noise (mean=0, std=1) to every element
torch::Tensor add_noise(const torch::Tensor& x) {
    auto noise = torch::randn_like(x);
    return x + noise;
}

// Register the operator into a library called "ops_sim"
TORCH_LIBRARY(ops_sim, m) {
    m.def("add_one(Tensor x) -> Tensor");
    m.def("add_noise(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(ops_sim, CPU, m) {
    m.impl("add_one", torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(add_one)));
    m.impl("add_noise", torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(add_noise)));
}