//
// Created by ang on 2021/1/13.
//

#include "GatherElements.h"
#include <cassert>
#include <stdio.h>

void onnx2trt::GatherElementsPlugin::deserialize(void const *data, size_t length) {
    const char *d = reinterpret_cast<const char *>(data);
    read(d, dim_);
//    read(d, t_c_);
//    read(d, t_h_);
//    read(d, t_w_);
//    read(d, idx_c_);
//    read(d, idx_h_);
//    read(d, idx_w_);
}

void onnx2trt::GatherElementsPlugin::serialize(void *buffer) const {
    char *d = static_cast<char *>(buffer);
    write(d, dim_);
//    write(d, t_c_);
//    write(d, t_h_);
//    write(d, t_w_);
//    write(d, idx_c_);
//    write(d, idx_h_);
//    write(d, idx_w_);
}

size_t onnx2trt::GatherElementsPlugin::getSerializationSize() const {
    return sizeof(dim_) + sizeof(t_h_) + sizeof(t_w_) + sizeof(idx_h_) + sizeof(idx_w_) + sizeof(t_c_) + sizeof(idx_c_);
}

//onnx2trt::GatherElementsPlugin::GatherElementsPlugin(
//        unsigned int dim, unsigned int t_c, unsigned int t_h, unsigned t_w, unsigned int idx_c, unsigned int idx_h, unsigned int idx_w) :
//        dim_(dim), t_c_(t_c), t_h_(t_h), t_w_(t_w), idx_c_(idx_c), idx_h_(idx_h), idx_w_(idx_w) {
//
//}
onnx2trt::GatherElementsPlugin::GatherElementsPlugin(
        unsigned int dim) :
        dim_(dim){
}

onnx2trt::GatherElementsPlugin::GatherElementsPlugin(void const *data, size_t length) {
    this->deserialize(data, length);
}

const char *onnx2trt::GatherElementsPlugin::getPluginType() const {
    return "GatherElements";
}

const char *onnx2trt::GatherElementsPlugin::getPluginVersion() const {
    return "1";
}

int onnx2trt::GatherElementsPlugin::getNbOutputs() const {
    return 1;
}

//Dims onnx2trt::GatherElementsPlugin::getOutputDimensions(int index, const Dims *inputs, int n_input_tensors) {
//    assert(index == 0 && n_input_tensors == 2 && inputs[0].nbDims < 4);
//    return inputs[1];
//}

//bool onnx2trt::GatherElementsPlugin::supportsFormat(DataType type, PluginFormat format) const {
//    return true;
//}

int onnx2trt::GatherElementsPlugin::initialize() { return 0; }

void onnx2trt::GatherElementsPlugin::terminate() {}

//size_t onnx2trt::GatherElementsPlugin::getWorkspaceSize(int32_t maxBatchSize) const {
//    return 0;
//}

void onnx2trt::GatherElementsPlugin::destroy() {
    delete this;
}

const char *onnx2trt::GatherElementsPlugin::getPluginNamespace() const {
    return "";
}

void onnx2trt::GatherElementsPlugin::setPluginNamespace(const char *N) {

}
//
//int onnx2trt::GatherElementsPlugin::enqueue(int batch_size, const void *const *inputs,
//                                  void **outputs, void *workspace, cudaStream_t stream) {
//    gather_elements(inputs, outputs, dim_, t_c_, t_h_, t_w_, idx_c_, idx_h_, idx_w_, stream);
//    return 0;
//}

// IPluginV2Ext Methods
DataType onnx2trt::GatherElementsPlugin::getOutputDataType(int index, const DataType *input_types, int n_inputs) const {
    assert(index == 0 && n_inputs == 2 && input_types[1] == DataType::kINT32);
    return input_types[0];
}

//bool onnx2trt::GatherElementsPlugin::isOutputBroadcastAcrossBatch(
//        int output_index, const bool *input_is_broadcast, int n_inputs) const {
//    return false;
//}

//bool onnx2trt::GatherElementsPlugin::canBroadcastInputAcrossBatch(int input_index) const { return false; }
//
//void onnx2trt::GatherElementsPlugin::configurePlugin(const Dims *input_dims, int n_inputs, const Dims *output_dims,
//                                           int n_outputs, const DataType *input_types, const DataType *output_types,
//                                           const bool *input_is_broadcast, const bool *output_is_broadcast,
//                                           PluginFormat float_format, int max_batch_size) {
//    assert(n_inputs == 1);
//    assert(input_dims != nullptr && input_dims[0].nbDims < 4);
//}

IPluginV2DynamicExt *onnx2trt::GatherElementsPlugin::clone() const {
//    return new GatherElementsPlugin(dim_, t_c_, t_h_, t_w_, idx_c_, idx_h_, idx_w_);
    return new GatherElementsPlugin(dim_);
}

size_t onnx2trt::GatherElementsPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs,
                                                        const PluginTensorDesc *outputs, int nbOutputs) const {
    return 0;
}

DimsExprs
onnx2trt::GatherElementsPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs,
                                                    IExprBuilder &exprBuilder) {

    assert(outputIndex == 0 && nbInputs == 2);
    return inputs[1];
}

bool
onnx2trt::GatherElementsPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs,
                                                          int32_t nbOutputs) {
    return true;
}

void onnx2trt::GatherElementsPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs,
                                                     const DynamicPluginTensorDesc *out, int32_t nbOutputs) {

    assert(nbInputs == 2 && nbOutputs == 1);
    assert(in[0].desc.dims.nbDims == in[1].desc.dims.nbDims);

    int dims = in[0].desc.dims.nbDims;

    if(1 == dims) {
        t_c_ = 1;
        t_h_ = 1;
        t_w_ = in[0].desc.dims.d[0];

        idx_c_ = 1;
        idx_w_ = 1;
        idx_h_ = in[1].desc.dims.d[0];
    }
    else if(2 == dims) {
        t_c_ = 1;
        t_h_ = in[0].desc.dims.d[0];
        t_w_ = in[0].desc.dims.d[1];

        idx_c_ = 1;
        idx_w_ = in[1].desc.dims.d[0];
        idx_h_ = in[1].desc.dims.d[1];
    }
    else {
        t_c_ = in[0].desc.dims.d[0];
        t_h_ = in[0].desc.dims.d[1];
        t_w_ = in[0].desc.dims.d[2];

        idx_c_ = in[1].desc.dims.d[0];
        idx_h_ = in[1].desc.dims.d[1];
        idx_w_ = in[1].desc.dims.d[2];
    }

}

//size_t onnx2trt::GatherElementsPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs,
//                                                        const PluginTensorDesc *outputs, int32_t nbOutputs) {
//    return 0;
//}

int32_t onnx2trt::GatherElementsPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc,
                                                const void *const *inputs, void *const *outputs, void *workspace,
                                                cudaStream_t stream) {

//    printf("indices type: %d\n", inputDesc[1].type);
//    printf("%d, %d, %d, %d, %d, %d\n", t_c_, t_h_, t_w_, idx_c_, idx_h_, idx_w_);
    gather_elements(inputs, outputs, dim_, t_c_, t_h_, t_w_, idx_c_, idx_h_, idx_w_, stream);
    return 0;
}

onnx2trt::GatherElementsPluginCreator::GatherElementsPluginCreator() = default;

const char* onnx2trt::GatherElementsPluginCreator::getPluginName() const {
    return "GatherElements";
}

const char* onnx2trt::GatherElementsPluginCreator::getPluginVersion() const {
    return "1";
}

const char* onnx2trt::GatherElementsPluginCreator::getPluginNamespace() const {
    return "";
}

IPluginV2* onnx2trt::GatherElementsPluginCreator::deserializePlugin(
        const char *name, const void *serial_data, size_t serial_length) {
    return new GatherElementsPlugin(serial_data, serial_length);
}

void onnx2trt::GatherElementsPluginCreator::setPluginNamespace(const char *N) {}

const PluginFieldCollection* onnx2trt::GatherElementsPluginCreator::getFieldNames() { return nullptr; }

IPluginV2* onnx2trt::GatherElementsPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) {
    return nullptr; }
