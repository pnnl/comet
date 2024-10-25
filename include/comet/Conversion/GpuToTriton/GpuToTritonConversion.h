//
// Copyright 2022 Battelle Memorial Institute
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions
// and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
// and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef GPU_DIALECT_TRITON_TRANSFORMS_COMETCONVERSION_H_
#define GPU_DIALECT_TRITON_TRANSFORMS_COMETCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace comet{

class GpuTypeConverter : public TypeConverter {
public:
  GpuTypeConverter(MLIRContext *context);
  int blockX, blockY, blockR;
private:
  MLIRContext *context;
};

class GpuConversionTarget : public ConversionTarget {

public:
  explicit GpuConversionTarget(MLIRContext &ctx,
                                     GpuTypeConverter &typeConverter);
};
class GpuTypeConverter2 : public TypeConverter {
public:
  GpuTypeConverter2(MLIRContext *context);
  int blockX, blockY, blockR;
private:
  MLIRContext *context;
};

class GpuConversionTarget2 : public ConversionTarget {

public:
  explicit GpuConversionTarget2(MLIRContext &ctx,
                                     GpuTypeConverter2 &typeConverter);
};

}
} // namespace mlir

#endif