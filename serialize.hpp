//
// Created by ang on 2021/1/13.
//

#ifndef ONNX2TRT_SERIALIZE_H
#define ONNX2TRT_SERIALIZE_H

template<typename T>
void write(char *&buffer, const T &val) {
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}

template<typename T>
void read(const char *&buffer, T &val) {
    val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
}

#endif //ONNX2TRT_SERIALIZE_H
