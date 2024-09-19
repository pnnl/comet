
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