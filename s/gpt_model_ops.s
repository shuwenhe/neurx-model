package neurx.gpt_model_ops

use neurx.tensor.tensor

func matmul(tensor a, tensor b) tensor {
    matmul(a, b)
}

func softmax(tensor a, int dim) tensor {
    softmax(a, dim)
}

func silu(tensor a) tensor {
    silu(a)
}

func gelu(tensor a, bool approximate) tensor {
    gelu(a, approximate)
}
