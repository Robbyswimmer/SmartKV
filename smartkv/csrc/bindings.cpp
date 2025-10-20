/*
 * SmartKV CUDA Extensions - PyTorch Bindings
 *
 * Python bindings for CUDA kernels.
 */

#include <torch/extension.h>
#include "quantized_attention.h"
#include "bit_packing.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Quantized attention
    m.def(
        "quantized_attention_forward",
        &quantized_attention_forward,
        "Fused quantized attention with on-the-fly dequantization (CUDA)",
        py::arg("query"),
        py::arg("key_int8"),
        py::arg("key_scale"),
        py::arg("value_int8"),
        py::arg("value_scale"),
        py::arg("attention_mask") = py::none()
    );

    m.def(
        "quantize_per_head_forward",
        &quantize_per_head_forward,
        "Per-head symmetric quantization (CUDA)",
        py::arg("input"),
        py::arg("bits")
    );

    m.def(
        "quantized_attention_bucket_forward",
        &quantized_attention_bucket_forward,
        "Bucket-aware fused quantized attention with on-the-fly unpacking (CUDA)",
        py::arg("query"),
        py::arg("key_qx"),
        py::arg("key_scale"),
        py::arg("value_qx"),
        py::arg("value_scale"),
        py::arg("global_slots"),
        py::arg("bits"),
        py::arg("packed_dim"),
        py::arg("is_packed"),
        py::arg("attention_mask") = py::none(),
        py::arg("full_k_len") = -1
    );

    // Bit-packing functions
    m.def(
        "pack_2bit",
        &pack_2bit,
        "Pack int8 values to 2-bit (4 values per byte)",
        py::arg("input")
    );

    m.def(
        "unpack_2bit",
        &unpack_2bit,
        "Unpack 2-bit values to int8",
        py::arg("packed"),
        py::arg("shape")
    );

    m.def(
        "pack_3bit",
        &pack_3bit,
        "Pack int8 values to 3-bit (8 values per 3 bytes)",
        py::arg("input")
    );

    m.def(
        "unpack_3bit",
        &unpack_3bit,
        "Unpack 3-bit values to int8",
        py::arg("packed"),
        py::arg("shape")
    );

    m.def(
        "pack_4bit",
        &pack_4bit,
        "Pack int8 values to 4-bit (2 values per byte)",
        py::arg("input")
    );

    m.def(
        "unpack_4bit",
        &unpack_4bit,
        "Unpack 4-bit values to int8",
        py::arg("packed"),
        py::arg("shape")
    );

    m.def(
        "pack_values",
        &pack_values,
        "Generic pack function (dispatches based on bits)",
        py::arg("input"),
        py::arg("bits")
    );

    m.def(
        "unpack_values",
        &unpack_values,
        "Generic unpack function (dispatches based on bits)",
        py::arg("packed"),
        py::arg("bits"),
        py::arg("shape")
    );
}
