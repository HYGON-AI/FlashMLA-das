#include <pybind11/pybind11.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_decode_qkvfp8.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("dense_decode_fwd_qkvfp8", &dense_attn_decode_qkvfp8_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
}
