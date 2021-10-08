// #define TEMPLATE(T)                                                          \
//     static inline void OPEN3D_SPECIALIZED(T, CPUAddElement)(                 \
//             int64_t idx, uniform Indexer * uniform indexer) {                \
//         const T* lhs =                                                       \
//                 OPEN3D_SPECIALIZED(T, Indexer_GetInputPtr)(indexer, 0, idx); \
//         const T* rhs =                                                       \
//                 OPEN3D_SPECIALIZED(T, Indexer_GetInputPtr)(indexer, 1, idx); \
//         T* dst = OPEN3D_SPECIALIZED(T, Indexer_GetOutputPtr)(indexer, idx);  \
//         *dst = *lhs + *rhs;                                                  \
//     }
// #pragma ignore warning(perf)
// OPEN3D_INSTANTIATE_TEMPLATE()
// #undef TEMPLATE

static inline void GetPtr(varying float** uniform input_ptr) {
    *input_ptr = NULL;
}

static inline void CPUAddElement_New() {
    varying float* varying lhs;
    varying float** uniform lhs_ref = (varying float** uniform)(&lhs);
    GetPtr(lhs_ref);
}

#define TEMPLATE(T)                                               \
    static inline void OPEN3D_SPECIALIZED(T, CPUAddElement)(      \
            int64_t idx, uniform Indexer * uniform indexer) {     \
        BinyEWPtrs biny_ew_ptrs = OPEN3D_SPECIALIZED(             \
                T, Indexer_GetInputOutputPtrs_2_1)(indexer, idx); \
    }
#pragma ignore warning(perf)
OPEN3D_INSTANTIATE_TEMPLATE()
#undef TEMPLATE

#define TEMPLATE(T)                                                          \
    static inline void OPEN3D_SPECIALIZED(T, CPUSubElement)(                 \
            int64_t idx, uniform Indexer * uniform indexer) {                \
        const T* lhs =                                                       \
                OPEN3D_SPECIALIZED(T, Indexer_GetInputPtr)(indexer, 0, idx); \
        const T* rhs =                                                       \
                OPEN3D_SPECIALIZED(T, Indexer_GetInputPtr)(indexer, 1, idx); \
        T* dst = OPEN3D_SPECIALIZED(T, Indexer_GetOutputPtr)(indexer, idx);  \
        *dst = *lhs - *rhs;                                                  \
    }
#pragma ignore warning(perf)
OPEN3D_INSTANTIATE_TEMPLATE()
#undef TEMPLATE

struct BinyEWPtrs {
    uint8_t* lhs_;
    uint8_t* rhs_;
    uint8_t* out_;
};

/// Get data pointer from a TensorRef with \p workload_idx.
/// Note: can be optimized by computing all input ptrs and output ptr
/// together.
#define TEMPLATE(T)                                                      \
    static inline BinyEWPtrs OPEN3D_SPECIALIZED(                         \
            T, Indexer_GetWorkloadDataPtrs_3)(                           \
            const uniform Indexer* const uniform self,                   \
            const uniform TensorRef* const uniform tr_0,                 \
            const uniform TensorRef* const uniform tr_1,                 \
            const uniform TensorRef* const uniform tr_2,                 \
            uniform bool tr_0_contiguous, uniform bool tr_1_contiguous,  \
            uniform bool tr_2_contiguous, int64_t workload_idx) {        \
        cif(workload_idx < 0) { return; }                                \
        if (tr_0_contiguous && tr_1_contiguous && tr_2_contiguous) {     \
            BinyEWPtrs ptrs;                                             \
            ptrs.lhs_ = (tr_0->data_ptr_) + workload_idx;                \
            ptrs.rhs_ = (tr_1->data_ptr_) + workload_idx;                \
            ptrs.out_ = (tr_2->data_ptr_) + workload_idx;                \
        } else {                                                         \
            int64_t offset_0 = 0;                                        \
            int64_t offset_1 = 0;                                        \
            int64_t offset_2 = 0;                                        \
            for (uniform int64_t i = 0; i < self->ndims_; ++i) {         \
                int64_t steps = workload_idx / self->master_strides_[i]; \
                offset_0 += steps * tr_0->byte_strides_[i];              \
                offset_1 += steps * tr_1->byte_strides_[i];              \
                offset_2 += steps * tr_2->byte_strides_[i];              \
                workload_idx = workload_idx % self->master_strides_[i];  \
            }                                                            \
            BinyEWPtrs ptrs;                                             \
            ptrs.lhs_ = ((uint8_t*)(tr_0->data_ptr_) + offset_0);        \
            ptrs.rhs_ = ((uint8_t*)(tr_1->data_ptr_) + offset_1);        \
            ptrs.out_ = ((uint8_t*)(tr_2->data_ptr_) + offset_2);        \
        }
#pragma ignore warning(perf)
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE

/// Get input Tensor data pointer based on \p workload_idx.
///
/// \param self The indexer instance.
/// \param input_idx Input tensor index.
/// \param workload_idx The index of the compute workload, similar to
/// thread_id, if a thread only processes one workload.
#define TEMPLATE(T)                                                            \
    static inline BinyEWPtrs OPEN3D_SPECIALIZED(                               \
            T, Indexer_GetInputOutputPtrs_2_1)(                                \
            const uniform Indexer* const uniform self, int64_t workload_idx) { \
        return OPEN3D_SPECIALIZED(T, Indexer_GetWorkloadDataPtrs_3)(           \
                self, &(self->inputs_[0]), &(self->inputs_[1]),                \
                &(self->outputs_[0]), self->inputs_contiguous_[0],             \
                self->inputs_contiguous_[1], self->outputs_contiguous_[0],     \
                workload_idx);                                                 \
    }
#pragma ignore warning(perf)
OPEN3D_INSTANTIATE_TEMPLATE_WITH_BOOL()
#undef TEMPLATE
