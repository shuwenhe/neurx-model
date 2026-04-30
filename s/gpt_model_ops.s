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

func attention(tensor q, tensor k, tensor v, tensor mask, float scale) tensor {
    tensor attn_scores = matmul(q, transpose(k))
    attn_scores = attn_scores * scale
    if mask != nil {
        attn_scores = where(mask == 0, -1e10, attn_scores)
    }
    tensor attn_weights = softmax(attn_scores, -1)
    tensor attn = matmul(attn_weights, v)
    return attn
}

func mlp(tensor x, tensor w1, tensor b1, tensor w2, tensor b2, string act) tensor {
    tensor h = matmul(x, w1)
    if b1 != nil {
        h = add(h, b1)
    }
    if act == "gelu" {
        h = gelu(h, false)
    } else if act == "silu" {
        h = silu(h)
    } else {
        h = relu(h)
    }
    tensor out = matmul(h, w2)
    if b2 != nil {
        out = add(out, b2)
    }
    return out
}

func block(tensor x, tensor ln1_w, tensor ln1_b, tensor attn_wq, tensor attn_wk, tensor attn_wv, tensor attn_wo, tensor mask, float scale, tensor ln2_w, tensor ln2_b, tensor mlp_w1, tensor mlp_b1, tensor mlp_w2, tensor mlp_b2, string act) tensor {
    tensor x1 = layer_norm(x, ln1_w, ln1_b)
    tensor q = matmul(x1, attn_wq)
    tensor k = matmul(x1, attn_wk)
    tensor v = matmul(x1, attn_wv)
    tensor attn_out = attention(q, k, v, mask, scale)
    tensor attn_proj = matmul(attn_out, attn_wo)
    tensor x2 = add(x, attn_proj)
    tensor x3 = layer_norm(x2, ln2_w, ln2_b)
    tensor mlp_out = mlp(x3, mlp_w1, mlp_b1, mlp_w2, mlp_b2, act)
    tensor out = add(x2, mlp_out)
    return out
}

func gpt_forward(tensor x, tensor[][] block_params, int n_layer, string act) tensor {
    tensor out = x
    int i = 0
    while i < n_layer {
        tensor[] p = block_params[i]
        out = block(out, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], act)
        i = i + 1
    }
    return out
}
