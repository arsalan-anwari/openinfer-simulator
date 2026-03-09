#ifndef OPENINFER_MATMUL_COMMON_SLANG
#define OPENINFER_MATMUL_COMMON_SLANG 1

struct MatmulPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
    uint acc_flags;  // encoded accumulation rule (dtype_code per byte)
};

#endif // OPENINFER_MATMUL_COMMON_SLANG
