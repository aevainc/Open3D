// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

// Contains source code from: https://github.com/rogersce/cnpy.
//
// The MIT License
//
// Copyright (c) Carl Rogers, 2011
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "open3d/t/io/NumpyIO.h"

#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

/////////////////
#include <stdint.h>

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <regex>
#include <stdexcept>
//////////////////

#include "open3d/core/Blob.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace io {

static char BigEndianChar() {
    int x = 1;
    return ((reinterpret_cast<char*>(&x))[0]) ? '<' : '>';
}

static char DtypeToChar(const core::Dtype& dtype) {
    // Not all dtypes are supported.
    // 'f': float, double, long double
    // 'i': int, char, short, long, long long
    // 'u': unsigned char, unsigned short, unsigned long, unsigned long long,
    //      unsigned int
    // 'b': bool
    // 'c': std::complex<float>, std::complex<double>),
    //      std::complex<long double>)
    // '?': object
    if (dtype == core::Float32) return 'f';
    if (dtype == core::Float64) return 'f';
    if (dtype == core::Int8) return 'i';
    if (dtype == core::Int16) return 'i';
    if (dtype == core::Int32) return 'i';
    if (dtype == core::Int64) return 'i';
    if (dtype == core::UInt8) return 'u';
    if (dtype == core::UInt16) return 'u';
    if (dtype == core::UInt32) return 'u';
    if (dtype == core::UInt64) return 'u';
    if (dtype == core::Bool) return 'b';
    utility::LogError("Unsupported dtype: {}", dtype.ToString());
    return '\0';
}

template <typename T>
static std::string ToByteString(const T& rhs) {
    std::stringstream ss;
    for (size_t i = 0; i < sizeof(T); i++) {
        char val = *(reinterpret_cast<const char*>(&rhs) + i);
        ss << val;
    }
    return ss.str();
}

static std::vector<char> CreateNumpyHeader(const core::SizeVector& shape,
                                           const core::Dtype& dtype) {
    // {}     -> "()"
    // {1}    -> "(1,)"
    // {1, 2} -> "(1, 2)"
    std::stringstream shape_ss;
    if (shape.size() == 0) {
        shape_ss << "()";
    } else if (shape.size() == 1) {
        shape_ss << fmt::format("({},)", shape[0]);
    } else {
        shape_ss << "(";
        shape_ss << shape[0];
        for (size_t i = 1; i < shape.size(); i++) {
            shape_ss << ", ";
            shape_ss << shape[i];
        }
        if (shape.size() == 1) {
            shape_ss << ",";
        }
        shape_ss << ")";
    }

    // Pad with spaces so that preamble+dict is modulo 16 bytes.
    // - Preamble is 10 bytes.
    // - Dict needs to end with '\n'.
    // - Header dict size includes the padding size and '\n'.
    std::string dict = fmt::format(
            "{{'descr': '{}{}{}', 'fortran_order': False, 'shape': {}, }}",
            BigEndianChar(), DtypeToChar(dtype), dtype.ByteSize(),
            shape_ss.str());
    size_t space_padding = 16 - (10 + dict.size()) % 16 - 1;  // {0, 1, ..., 15}
    dict.insert(dict.end(), space_padding, ' ');
    dict += '\n';

    std::stringstream ss;
    // "Magic" values.
    ss << (char)0x93;
    ss << "NUMPY";
    // Major version of numpy format.
    ss << (char)0x01;
    // Minor version of numpy format.
    ss << (char)0x00;
    // Header dict size (full header size - 10).
    ss << ToByteString((uint16_t)dict.size());
    // Header dict.
    ss << dict;

    std::string s = ss.str();
    return std::vector<char>(s.begin(), s.end());
}

static std::tuple<char, int64_t, core::SizeVector, bool> ParseNumpyHeader(
        FILE* fp) {
    char type;
    int64_t word_size;
    core::SizeVector shape;
    bool fortran_order;

    char buffer[256];
    size_t res = fread(buffer, sizeof(char), 11, fp);
    if (res != 11) {
        utility::LogError("Failed fread.");
    }
    std::string header;
    if (const char* header_chars = fgets(buffer, 256, fp)) {
        header = std::string(header_chars);
    } else {
        utility::LogError(
                "Numpy file header could not be read. "
                "Possibly the file is corrupted.");
    }
    if (header[header.size() - 1] != '\n') {
        utility::LogError("The last char must be '\n'.");
    }

    size_t loc1, loc2;

    // Fortran order.
    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos) {
        utility::LogError("Failed to find header keyword: 'fortran_order'");
    }

    loc1 += 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    // Shape.
    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos) {
        utility::LogError("Failed to find header keyword: '(' or ')'");
    }

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    // Endian, word size, data type.
    // byte order code | stands for not applicable.
    // not sure when this applies except for byte array.
    loc1 = header.find("descr");
    if (loc1 == std::string::npos) {
        utility::LogError("Failed to find header keyword: 'descr'");
    }

    loc1 += 9;
    bool little_endian =
            (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    if (!little_endian) {
        utility::LogError("Only big endian is supported.");
    }

    type = header[loc1 + 1];

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0, loc2).c_str());

    return std::make_tuple(type, word_size, shape, fortran_order);
}

class NumpyArray {
public:
    NumpyArray(const core::Tensor& t)
        : shape_(t.GetShape()),
          type_(DtypeToChar(t.GetDtype())),
          word_size_(t.GetDtype().ByteSize()),
          fortran_order_(false),
          num_elements_(t.GetShape().NumElements()) {
        blob_ = t.To(core::Device("CPU:0")).Contiguous().GetBlob();
    }

    NumpyArray(const core::SizeVector& shape,
               char type,
               int64_t word_size,
               bool fortran_order)
        : shape_(shape),
          type_(type),
          word_size_(word_size),
          fortran_order_(fortran_order) {
        num_elements_ = 1;
        for (size_t i = 0; i < shape_.size(); i++) {
            num_elements_ *= shape_[i];
        }
        blob_ = std::make_shared<core::Blob>(num_elements_ * word_size_,
                                             core::Device("CPU:0"));
    }

    template <typename T>
    T* GetDataPtr() {
        return reinterpret_cast<T*>(blob_->GetDataPtr());
    }

    template <typename T>
    const T* GetDataPtr() const {
        return reinterpret_cast<const T*>(blob_->GetDataPtr());
    }

    core::Dtype GetDtype() const {
        if (type_ == 'f' && word_size_ == 4) return core::Float32;
        if (type_ == 'f' && word_size_ == 8) return core::Float64;
        if (type_ == 'i' && word_size_ == 1) return core::Int8;
        if (type_ == 'i' && word_size_ == 2) return core::Int16;
        if (type_ == 'i' && word_size_ == 4) return core::Int32;
        if (type_ == 'i' && word_size_ == 8) return core::Int64;
        if (type_ == 'u' && word_size_ == 1) return core::UInt8;
        if (type_ == 'u' && word_size_ == 2) return core::UInt16;
        if (type_ == 'u' && word_size_ == 4) return core::UInt32;
        if (type_ == 'u' && word_size_ == 8) return core::UInt64;
        if (type_ == 'b') return core::Bool;

        return core::Undefined;
    }

    core::SizeVector GetShape() const { return shape_; }

    bool IsFortranOrder() const { return fortran_order_; }

    int64_t NumBytes() const { return num_elements_ * word_size_; }

    core::Tensor ToTensor() const {
        if (fortran_order_) {
            utility::LogError("Cannot load Numpy array with fortran_order.");
        }
        core::Dtype dtype = GetDtype();
        if (dtype.GetDtypeCode() == core::Dtype::DtypeCode::Undefined) {
            utility::LogError(
                    "Cannot load Numpy array with Numpy dtype={} and "
                    "word_size={}.",
                    type_, word_size_);
        }
        // t.blob_ is the same as blob_, no need for memory copy.
        core::Tensor t(shape_, core::shape_util::DefaultStrides(shape_),
                       const_cast<void*>(GetDataPtr<void>()), dtype, blob_);
        return t;
    }

    static NumpyArray CreateFromFile(const std::string& filename) {
        FILE* fp = fopen(filename.c_str(), "rb");
        if (!fp) {
            utility::LogError("Load: Unable to open file {}.", filename);
        }
        core::SizeVector shape;
        int64_t word_size;
        bool fortran_order;
        char type;
        std::tie(type, word_size, shape, fortran_order) = ParseNumpyHeader(fp);
        NumpyArray arr(shape, type, word_size, fortran_order);
        size_t nread = fread(arr.GetDataPtr<char>(), 1,
                             static_cast<size_t>(arr.NumBytes()), fp);
        if (nread != static_cast<size_t>(arr.NumBytes())) {
            utility::LogError("Load: failed fread");
        }
        fclose(fp);
        return arr;
    }

    void Save(std::string filename) const {
        FILE* fp = fopen(filename.c_str(), "wb");
        if (!fp) {
            utility::LogError("Save: Unable to open file {}.", filename);
            return;
        }
        std::vector<char> header = CreateNumpyHeader(shape_, GetDtype());
        fseek(fp, 0, SEEK_SET);
        fwrite(&header[0], sizeof(char), header.size(), fp);
        fseek(fp, 0, SEEK_END);
        fwrite(GetDataPtr<void>(), static_cast<size_t>(GetDtype().ByteSize()),
               static_cast<size_t>(shape_.NumElements()), fp);
        fclose(fp);
    }

private:
    std::shared_ptr<core::Blob> blob_ = nullptr;
    core::SizeVector shape_;
    char type_;
    int64_t word_size_;
    bool fortran_order_;
    int64_t num_elements_;
};

core::Tensor ReadNpy(const std::string& filename) {
    return NumpyArray::CreateFromFile(filename).ToTensor();
}

void WriteNpy(const std::string& filename, const core::Tensor& tensor) {
    NumpyArray(tensor).Save(filename);
}

std::unordered_map<std::string, core::Tensor> ReadNpz(
        const std::string& filename) {
    return {};
}

void WriteNpz(const std::string& filename,
              const std::unordered_map<std::string, core::Tensor>& tensor_map) {
}

}  // namespace io
}  // namespace t
}  // namespace open3d

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace cnpy {

struct NpyArray {
    NpyArray(const std::vector<size_t>& _shape,
             size_t _word_size,
             bool _fortran_order)
        : shape(_shape), word_size(_word_size), fortran_order(_fortran_order) {
        num_vals = 1;
        for (size_t i = 0; i < shape.size(); i++) num_vals *= shape[i];
        data_holder = std::shared_ptr<std::vector<char>>(
                new std::vector<char>(num_vals * word_size));
    }

    NpyArray() : shape(0), word_size(0), fortran_order(0), num_vals(0) {}

    template <typename T>
    T* data() {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }

    template <typename T>
    const T* data() const {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }

    template <typename T>
    std::vector<T> as_vec() const {
        const T* p = data<T>();
        return std::vector<T>(p, p + num_vals);
    }

    size_t num_bytes() const { return data_holder->size(); }

    std::shared_ptr<std::vector<char>> data_holder;
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    size_t num_vals;
};

char BigEndianTest();

char map_type(const std::type_info& t);

template <typename T>
std::vector<char> create_npy_header(const std::vector<size_t>& shape);

void parse_npy_header(FILE* fp,
                      size_t& word_size,
                      std::vector<size_t>& shape,
                      bool& fortran_order);

void parse_npy_header(unsigned char* buffer,
                      size_t& word_size,
                      std::vector<size_t>& shape,
                      bool& fortran_order);

void parse_zip_footer(FILE* fp,
                      uint16_t& nrecs,
                      size_t& global_header_size,
                      size_t& global_header_offset);

std::map<std::string, NpyArray> npz_load(std::string fname);

NpyArray npz_load(std::string fname, std::string varname);

template <typename T>
std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
    // write in little endian
    for (size_t byte = 0; byte < sizeof(T); byte++) {
        char val = *((char*)&rhs + byte);
        lhs.push_back(val);
    }
    return lhs;
}

template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs);

template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs);

template <typename T>
void npz_save(std::string zipname,
              std::string fname,
              const T* data,
              const std::vector<size_t>& shape,
              std::string mode = "w") {
    // first, append a .npy to the fname
    fname += ".npy";

    // now, on with the show
    FILE* fp = NULL;
    uint16_t nrecs = 0;
    size_t global_header_offset = 0;
    std::vector<char> global_header;

    if (mode == "a") fp = fopen(zipname.c_str(), "r+b");

    if (fp) {
        // zip file exists. we need to add a new npy file to it.
        // first read the footer. this gives us the offset and size of the
        // global header then read and store the global header. below, we will
        // write the the new data at the start of the global header then append
        // the global header and footer below it
        size_t global_header_size;
        parse_zip_footer(fp, nrecs, global_header_size, global_header_offset);
        fseek(fp, global_header_offset, SEEK_SET);
        global_header.resize(global_header_size);
        size_t res =
                fread(&global_header[0], sizeof(char), global_header_size, fp);
        if (res != global_header_size) {
            throw std::runtime_error(
                    "npz_save: header read error while adding to existing zip");
        }
        fseek(fp, global_header_offset, SEEK_SET);
    } else {
        fp = fopen(zipname.c_str(), "wb");
    }

    std::vector<char> npy_header = create_npy_header<T>(shape);

    size_t nels = std::accumulate(shape.begin(), shape.end(), 1,
                                  std::multiplies<size_t>());
    size_t nbytes = nels * sizeof(T) + npy_header.size();

    // get the CRC of the data to be added
    uint32_t crc = crc32(0L, (uint8_t*)&npy_header[0], npy_header.size());
    crc = crc32(crc, (uint8_t*)data, nels * sizeof(T));

    // build the local header
    std::vector<char> local_header;
    local_header += "PK";                    // first part of sig
    local_header += (uint16_t)0x0403;        // second part of sig
    local_header += (uint16_t)20;            // min version to extract
    local_header += (uint16_t)0;             // general purpose bit flag
    local_header += (uint16_t)0;             // compression method
    local_header += (uint16_t)0;             // file last mod time
    local_header += (uint16_t)0;             // file last mod date
    local_header += (uint32_t)crc;           // crc
    local_header += (uint32_t)nbytes;        // compressed size
    local_header += (uint32_t)nbytes;        // uncompressed size
    local_header += (uint16_t)fname.size();  // fname length
    local_header += (uint16_t)0;             // extra field length
    local_header += fname;

    // build global header
    global_header += "PK";              // first part of sig
    global_header += (uint16_t)0x0201;  // second part of sig
    global_header += (uint16_t)20;      // version made by
    global_header.insert(global_header.end(), local_header.begin() + 4,
                         local_header.begin() + 30);
    global_header += (uint16_t)0;  // file comment length
    global_header += (uint16_t)0;  // disk number where file starts
    global_header += (uint16_t)0;  // internal file attributes
    global_header += (uint32_t)0;  // external file attributes
    global_header +=
            (uint32_t)global_header_offset;  // relative offset of local file
                                             // header, since it begins where
                                             // the global header used to begin
    global_header += fname;

    // build footer
    std::vector<char> footer;
    footer += "PK";                            // first part of sig
    footer += (uint16_t)0x0605;                // second part of sig
    footer += (uint16_t)0;                     // number of this disk
    footer += (uint16_t)0;                     // disk where footer starts
    footer += (uint16_t)(nrecs + 1);           // number of records on this disk
    footer += (uint16_t)(nrecs + 1);           // total number of records
    footer += (uint32_t)global_header.size();  // nbytes of global headers
    footer += (uint32_t)(
            global_header_offset + nbytes +
            local_header.size());  // offset of start of global
                                   // headers, since global header now
                                   // starts after newly written array
    footer += (uint16_t)0;         // zip file comment length

    // write everything
    fwrite(&local_header[0], sizeof(char), local_header.size(), fp);
    fwrite(&npy_header[0], sizeof(char), npy_header.size(), fp);
    fwrite(data, sizeof(T), nels, fp);
    fwrite(&global_header[0], sizeof(char), global_header.size(), fp);
    fwrite(&footer[0], sizeof(char), footer.size(), fp);
    fclose(fp);
}

template <typename T>
void npz_save(std::string zipname,
              std::string fname,
              const std::vector<T> data,
              std::string mode = "w") {
    std::vector<size_t> shape;
    shape.push_back(data.size());
    npz_save(zipname, fname, &data[0], shape, mode);
}

template <typename T>
std::vector<char> create_npy_header(const std::vector<size_t>& shape) {
    std::vector<char> dict;
    dict += "{'descr': '";
    dict += BigEndianTest();
    dict += map_type(typeid(T));
    dict += std::to_string(sizeof(T));
    dict += "', 'fortran_order': False, 'shape': (";
    dict += std::to_string(shape[0]);
    for (size_t i = 1; i < shape.size(); i++) {
        dict += ", ";
        dict += std::to_string(shape[i]);
    }
    if (shape.size() == 1) dict += ",";
    dict += "), }";
    // pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10
    // bytes. dict needs to end with \n
    int remainder = 16 - (10 + dict.size()) % 16;
    dict.insert(dict.end(), remainder, ' ');
    dict.back() = '\n';

    std::vector<char> header;
    header += (char)0x93;
    header += "NUMPY";
    header += (char)0x01;  // major version of numpy format
    header += (char)0x00;  // minor version of numpy format
    header += (uint16_t)dict.size();
    header.insert(header.end(), dict.begin(), dict.end());

    return header;
}

char BigEndianTest() {
    int x = 1;
    return (((char*)&x)[0]) ? '<' : '>';
}

char map_type(const std::type_info& t) {
    if (t == typeid(float)) return 'f';
    if (t == typeid(double)) return 'f';
    if (t == typeid(long double)) return 'f';

    if (t == typeid(int)) return 'i';
    if (t == typeid(char)) return 'i';
    if (t == typeid(short)) return 'i';
    if (t == typeid(long)) return 'i';
    if (t == typeid(long long)) return 'i';

    if (t == typeid(unsigned char)) return 'u';
    if (t == typeid(unsigned short)) return 'u';
    if (t == typeid(unsigned long)) return 'u';
    if (t == typeid(unsigned long long)) return 'u';
    if (t == typeid(unsigned int)) return 'u';

    if (t == typeid(bool)) return 'b';

    if (t == typeid(std::complex<float>)) return 'c';
    if (t == typeid(std::complex<double>)) return 'c';
    if (t == typeid(std::complex<long double>))
        return 'c';

    else
        return '?';
}

template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs) {
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return lhs;
}

template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs) {
    // write in little endian
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for (size_t byte = 0; byte < len; byte++) {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}

void parse_npy_header(unsigned char* buffer,
                      size_t& word_size,
                      std::vector<size_t>& shape,
                      bool& fortran_order) {
    // std::string magic_string(buffer,6);
    uint8_t major_version = *reinterpret_cast<uint8_t*>(buffer + 6);
    (void)major_version;
    uint8_t minor_version = *reinterpret_cast<uint8_t*>(buffer + 7);
    (void)minor_version;
    uint16_t header_len = *reinterpret_cast<uint16_t*>(buffer + 8);
    std::string header(reinterpret_cast<char*>(buffer + 9), header_len);

    size_t loc1, loc2;

    // fortran order
    loc1 = header.find("fortran_order") + 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    // shape
    loc1 = header.find("(");
    loc2 = header.find(")");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    // endian, word size, data type
    // byte order code | stands for not applicable.
    // not sure when this applies except for byte array
    loc1 = header.find("descr") + 9;
    bool littleEndian =
            (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);
    (void)littleEndian;

    // char type = header[loc1+1];
    // assert(type == map_type(T));

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0, loc2).c_str());
}

void parse_npy_header(FILE* fp,
                      size_t& word_size,
                      std::vector<size_t>& shape,
                      bool& fortran_order) {
    char buffer[256];
    size_t res = fread(buffer, sizeof(char), 11, fp);
    if (res != 11) throw std::runtime_error("parse_npy_header: failed fread");
    std::string header = fgets(buffer, 256, fp);
    assert(header[header.size() - 1] == '\n');

    size_t loc1, loc2;

    // fortran order
    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error(
                "parse_npy_header: failed to find header keyword: "
                "'fortran_order'");
    loc1 += 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    // shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error(
                "parse_npy_header: failed to find header keyword: '(' or ')'");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    // endian, word size, data type
    // byte order code | stands for not applicable.
    // not sure when this applies except for byte array
    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error(
                "parse_npy_header: failed to find header keyword: 'descr'");
    loc1 += 9;
    bool littleEndian =
            (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);
    (void)littleEndian;

    // char type = header[loc1+1];
    // assert(type == map_type(T));

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0, loc2).c_str());
}

void parse_zip_footer(FILE* fp,
                      uint16_t& nrecs,
                      size_t& global_header_size,
                      size_t& global_header_offset) {
    std::vector<char> footer(22);
    fseek(fp, -22, SEEK_END);
    size_t res = fread(&footer[0], sizeof(char), 22, fp);
    if (res != 22) throw std::runtime_error("parse_zip_footer: failed fread");

    uint16_t disk_no, disk_start, nrecs_on_disk, comment_len;
    disk_no = *(uint16_t*)&footer[4];
    disk_start = *(uint16_t*)&footer[6];
    nrecs_on_disk = *(uint16_t*)&footer[8];
    nrecs = *(uint16_t*)&footer[10];
    global_header_size = *(uint32_t*)&footer[12];
    global_header_offset = *(uint32_t*)&footer[16];
    comment_len = *(uint16_t*)&footer[20];

    assert(disk_no == 0);
    assert(disk_start == 0);
    assert(nrecs_on_disk == nrecs);
    assert(comment_len == 0);
    (void)disk_no;
    (void)disk_start;
    (void)nrecs_on_disk;
    (void)comment_len;
}

NpyArray load_the_npy_file(FILE* fp) {
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    parse_npy_header(fp, word_size, shape, fortran_order);

    NpyArray arr(shape, word_size, fortran_order);
    size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
    if (nread != arr.num_bytes())
        throw std::runtime_error("load_the_npy_file: failed fread");
    return arr;
}

NpyArray load_the_npz_array(FILE* fp,
                            uint32_t compr_bytes,
                            uint32_t uncompr_bytes) {
    std::vector<unsigned char> buffer_compr(compr_bytes);
    std::vector<unsigned char> buffer_uncompr(uncompr_bytes);
    size_t nread = fread(&buffer_compr[0], 1, compr_bytes, fp);
    if (nread != compr_bytes)
        throw std::runtime_error("load_the_npy_file: failed fread");

    int err;
    z_stream d_stream;

    d_stream.zalloc = Z_NULL;
    d_stream.zfree = Z_NULL;
    d_stream.opaque = Z_NULL;
    d_stream.avail_in = 0;
    d_stream.next_in = Z_NULL;
    err = inflateInit2(&d_stream, -MAX_WBITS);

    d_stream.avail_in = compr_bytes;
    d_stream.next_in = &buffer_compr[0];
    d_stream.avail_out = uncompr_bytes;
    d_stream.next_out = &buffer_uncompr[0];

    err = inflate(&d_stream, Z_FINISH);
    err = inflateEnd(&d_stream);
    (void)err;

    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    parse_npy_header(&buffer_uncompr[0], word_size, shape, fortran_order);

    NpyArray array(shape, word_size, fortran_order);

    size_t offset = uncompr_bytes - array.num_bytes();
    memcpy(array.data<unsigned char>(), &buffer_uncompr[0] + offset,
           array.num_bytes());

    return array;
}

std::map<std::string, NpyArray> npz_load(std::string fname) {
    FILE* fp = fopen(fname.c_str(), "rb");

    if (!fp) {
        throw std::runtime_error("npz_load: Error! Unable to open file " +
                                 fname + "!");
    }

    std::map<std::string, NpyArray> arrays;

    while (1) {
        std::vector<char> local_header(30);
        size_t headerres = fread(&local_header[0], sizeof(char), 30, fp);
        if (headerres != 30) throw std::runtime_error("npz_load: failed fread");

        // if we've reached the global header, stop reading
        if (local_header[2] != 0x03 || local_header[3] != 0x04) break;

        // read in the variable name
        uint16_t name_len = *(uint16_t*)&local_header[26];
        std::string varname(name_len, ' ');
        size_t vname_res = fread(&varname[0], sizeof(char), name_len, fp);
        if (vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");

        // erase the lagging .npy
        varname.erase(varname.end() - 4, varname.end());

        // read in the extra field
        uint16_t extra_field_len = *(uint16_t*)&local_header[28];
        if (extra_field_len > 0) {
            std::vector<char> buff(extra_field_len);
            size_t efield_res =
                    fread(&buff[0], sizeof(char), extra_field_len, fp);
            if (efield_res != extra_field_len)
                throw std::runtime_error("npz_load: failed fread");
        }

        uint16_t compr_method =
                *reinterpret_cast<uint16_t*>(&local_header[0] + 8);
        uint32_t compr_bytes =
                *reinterpret_cast<uint32_t*>(&local_header[0] + 18);
        uint32_t uncompr_bytes =
                *reinterpret_cast<uint32_t*>(&local_header[0] + 22);

        if (compr_method == 0) {
            arrays[varname] = load_the_npy_file(fp);
        } else {
            arrays[varname] =
                    load_the_npz_array(fp, compr_bytes, uncompr_bytes);
        }
    }

    fclose(fp);
    return arrays;
}

NpyArray npz_load(std::string fname, std::string varname) {
    FILE* fp = fopen(fname.c_str(), "rb");

    if (!fp) throw std::runtime_error("npz_load: Unable to open file " + fname);

    while (1) {
        std::vector<char> local_header(30);
        size_t header_res = fread(&local_header[0], sizeof(char), 30, fp);
        if (header_res != 30)
            throw std::runtime_error("npz_load: failed fread");

        // if we've reached the global header, stop reading
        if (local_header[2] != 0x03 || local_header[3] != 0x04) break;

        // read in the variable name
        uint16_t name_len = *(uint16_t*)&local_header[26];
        std::string vname(name_len, ' ');
        size_t vname_res = fread(&vname[0], sizeof(char), name_len, fp);
        if (vname_res != name_len)
            throw std::runtime_error("npz_load: failed fread");
        vname.erase(vname.end() - 4, vname.end());  // erase the lagging .npy

        // read in the extra field
        uint16_t extra_field_len = *(uint16_t*)&local_header[28];
        fseek(fp, extra_field_len, SEEK_CUR);  // skip past the extra field

        uint16_t compr_method =
                *reinterpret_cast<uint16_t*>(&local_header[0] + 8);
        uint32_t compr_bytes =
                *reinterpret_cast<uint32_t*>(&local_header[0] + 18);
        uint32_t uncompr_bytes =
                *reinterpret_cast<uint32_t*>(&local_header[0] + 22);

        if (vname == varname) {
            NpyArray array = (compr_method == 0)
                                     ? load_the_npy_file(fp)
                                     : load_the_npz_array(fp, compr_bytes,
                                                          uncompr_bytes);
            fclose(fp);
            return array;
        } else {
            // skip past the data
            uint32_t size = *(uint32_t*)&local_header[22];
            fseek(fp, size, SEEK_CUR);
        }
    }

    fclose(fp);

    // if we get here, we haven't found the variable in the file
    throw std::runtime_error("npz_load: Variable name " + varname +
                             " not found in " + fname);
}

void CnpyIOTest() {
    const int Nx = 128;
    const int Ny = 64;
    const int Nz = 32;

    // set random seed so that result is reproducible (for testing)
    srand(0);
    // create random data
    std::vector<std::complex<double>> data(Nx * Ny * Nz);
    for (int i = 0; i < Nx * Ny * Nz; i++)
        data[i] = std::complex<double>(rand(), rand());

    // now write to an npz file
    // non-array variables are treated as 1D arrays with 1 element
    double myVar1 = 1.2;
    char myVar2 = 'a';
    npz_save("out.npz", "myVar1", &myVar1, {1},
             "w");  //"w" overwrites any existing file
    npz_save("out.npz", "myVar2", &myVar2, {1},
             "a");  //"a" appends to the file we created above
    npz_save("out.npz", "arr1", &data[0], {Nz, Ny, Nx},
             "a");  //"a" appends to the file we created above

    // load a single var from the npz file
    NpyArray arr2 = npz_load("out.npz", "arr1");

    // load the entire npz file
    std::map<std::string, NpyArray> my_npz = npz_load("out.npz");

    // check that the loaded myVar1 matches myVar1
    NpyArray arr_mv1 = my_npz["myVar1"];
    double* mv1 = arr_mv1.data<double>();
    (void)mv1;
    assert(arr_mv1.shape.size() == 1 && arr_mv1.shape[0] == 1);
    assert(mv1[0] == myVar1);
}

}  // namespace cnpy
