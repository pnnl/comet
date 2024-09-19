#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "comet/Conversion/GpuToTriton/GpuToTritonConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
using namespace mlir;


mlir::comet::GpuConversionTarget::GpuConversionTarget(
    MLIRContext &context, GpuTypeConverter &typeConverter)
    : ConversionTarget(context) {
  // TODO: we should also verify ops of GpuDialect
    addLegalDialect<arith::ArithDialect, triton::TritonDialect, mlir::BuiltinDialect, mlir::affine::AffineDialect>();
    addLegalOp<mlir::gpu::ReturnOp>();
    addIllegalDialect<memref::MemRefDialect>();

  // Some ops from SCF are illegal
//   addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
//                scf::ReduceReturnOp>();

  addDynamicallyLegalDialect<scf::SCFDialect>([&](Operation *op) {

    if (isa<scf::ForOp>(op)&& op->hasAttr("loop_block_size"))
    {
      return true;
    }
    else if (isa<scf::ForOp>(op)&& (op->hasAttr("programs_loop_x") || op->hasAttr("programs_loop_y")) )
    {
      return true;
    }
    else if(!isa<scf::ForOp>(op)) {
      return true;
    }

    return false;
  });

  addDynamicallyLegalDialect<memref::MemRefDialect>([&](Operation *op) {
    if (op->getParentOfType<gpu::GPUFuncOp>() || op->getParentOfType<triton::FuncOp>())
    {
      return false;
    }

    return true;

  });

  addDynamicallyLegalOp<affine::AffineApplyOp>([](affine::AffineApplyOp op) -> bool 
    {

      for(auto oper: op->getOperands())
      {
        if(llvm::isa_and_nonnull<affine::AffineApplyOp>(oper.getDefiningOp()))
        {
          return false;
        }
      }

      return true;
    });

  addDynamicallyLegalOp<mlir::gpu::GPUFuncOp>([](mlir::gpu::GPUFuncOp op) -> bool 
  {
    if(op->hasAttr("copied") || op.getFunctionBody().empty())
    {
      return true;
    }

    return false;
  });


  addDynamicallyLegalOp<scf::IfOp>([](scf::IfOp op) -> bool 
    {
      for(auto oper : op.getCondition().getDefiningOp()->getOperands())
      {
        if(!llvm::isa_and_nonnull<affine::AffineApplyOp>(oper.getDefiningOp()))
        {
          return false;
        }
      }

      return true;
    });
  // We have requirements for the data layouts
//   addDynamicallyLegalOp<triton::DotOp>([](triton::DotOp dotOp) -> bool {
//     Attribute aEncoding =
//         dotOp.getA().getType().cast<RankedTensorType>().getEncoding();
//     Attribute bEncoding =
//         dotOp.getB().getType().cast<RankedTensorType>().getEncoding();
//     if (aEncoding && aEncoding.isa<triton::gpu::DotOperandEncodingAttr>() &&
//         bEncoding && bEncoding.isa<triton::gpu::DotOperandEncodingAttr>())
//       return true;
//     return false;
//   });
}

mlir::comet::GpuConversionTarget2::GpuConversionTarget2(
    MLIRContext &context, GpuTypeConverter2 &typeConverter)
    : ConversionTarget(context) {
  // TODO: we should also verify ops of GpuDialect
    addLegalDialect<arith::ArithDialect, triton::TritonDialect, scf::SCFDialect, mlir::BuiltinDialect>();
    addIllegalDialect<memref::MemRefDialect>();

  // Some ops from SCF are illegal
//   addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
//                scf::ReduceReturnOp>();

  addDynamicallyLegalDialect<memref::MemRefDialect>([&](Operation *op) {
    if (op->getParentOfType<gpu::GPUFuncOp>() || op->getParentOfType<triton::FuncOp>())
    {
      return false;
    }

    return true;

  });
  
  addDynamicallyLegalOp<arith::AddFOp>([](arith::AddFOp op) -> bool 
    {
      for(auto opr: op->getOperands())
      {
        bool eq = op.getType() == opr.getType();
        if (!eq)
        {
          return false;
        }
      }

      return true;
    });

  addDynamicallyLegalOp<arith::CmpFOp>([](arith::CmpFOp op) -> bool 
    {
      Type prev = NULL;
      for(auto opr: op->getOperands())
      {
        if(prev == NULL)
        {
          prev = opr.getType();
        }
        else if (prev != opr.getType())
        {
          return false;
        }
      }
      if(prev != op.getType())
      {
        if(!prev.isa<RankedTensorType>() && !op.getType().isa<RankedTensorType>())
        {
          return true;
        }
        else if(prev.isa<RankedTensorType>() && op.getType().isa<RankedTensorType>())
        {
          if(prev.cast<RankedTensorType>().getShape() == op.getType().cast<RankedTensorType>().getShape())
          {
            return true;
          }
          else 
          {
            return false;
          }
        }

        return false;
      }

      return true;
    });

  addDynamicallyLegalOp<arith::CmpIOp>([](arith::CmpIOp op) -> bool 
    {
      Type prev = NULL;
      for(auto opr: op->getOperands())
      {
        if(prev == NULL)
        {
          prev = opr.getType();
        }
        else if (prev != opr.getType())
        {
          return false;
        }
      }
      if(prev != op.getType())
      {
        if(!prev.isa<RankedTensorType>() && !op.getType().isa<RankedTensorType>())
        {
          return true;
        }
        else if(prev.isa<RankedTensorType>() && op.getType().isa<RankedTensorType>())
        {
          if(prev.cast<RankedTensorType>().getShape() == op.getType().cast<RankedTensorType>().getShape())
          {
            return true;
          }
          else 
          {
            return false;
          }
        }

        return false;
      }

      return true;
    });
    
  addDynamicallyLegalOp<arith::SelectOp>([](arith::SelectOp op) -> bool 
    {
      if(op.getTrueValue().getType() == op.getResult().getType() && op.getFalseValue().getType() == op.getTrueValue().getType())
      {
        if(op.getCondition().getType().isa<RankedTensorType>() && !op.getFalseValue().getType().isa<RankedTensorType>())
        {
          return false;
        }
        else if(!op.getCondition().getType().isa<RankedTensorType>() && op.getFalseValue().getType().isa<RankedTensorType>())
        {
          return false;
        }
        else if(op.getCondition().getType().isa<RankedTensorType>() && op.getFalseValue().getType().isa<RankedTensorType>())
        {
          if(op.getCondition().getType().cast<RankedTensorType>().getShape() == op.getFalseValue().getType().cast<RankedTensorType>().getShape())
          {
            return true;
          }
          else 
          {
            return false;
          }
        }
      }

      return false;
    });
  // addDynamicallyLegalOp<affine::AffineApplyOp>([](affine::AffineApplyOp op) -> bool 
  //   {

  //     for(auto oper: op->getOperands())
  //     {
  //       if(auto affineOp = llvm::isa_and_nonnull<affine::AffineApplyOp>(oper.getDefiningOp()))
  //       {
  //         return false;
  //       }
  //     }

  //     return true;
  //   });

  addDynamicallyLegalOp<arith::SubFOp>([](arith::SubFOp op) -> bool 
    {
      for(auto opr: op->getOperands())
      {
        bool eq = op.getType() == opr.getType();
        if (!eq)
        {
          return false;
        }
      }

      return true;
    });
    
  addDynamicallyLegalOp<arith::SubIOp>([](arith::SubIOp op) -> bool 
    {
      for(auto opr: op->getOperands())
      {
        bool eq = op.getType() == opr.getType();
        if (!eq)
        {
          return false;
        }
      }

      return true;
    });

  addDynamicallyLegalOp<arith::MulFOp>([](arith::MulFOp op) -> bool 
    {
      for(auto opr: op->getOperands())
      {
        bool eq = op.getType() == opr.getType();
        if (!eq)
        {
          return false;
        }
      }

      return true;
    });
  addDynamicallyLegalOp<arith::DivSIOp>([](arith::DivSIOp op) -> bool 
    {
      for(auto opr: op->getOperands())
      {
        bool eq = op.getType() == opr.getType();
        if (!eq)
        {
          return false;
        }
      }

      return true;
    });

  addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp op) -> bool 
    {
      for(auto opr: op.getArgumentTypes())
      {
        if(opr.isa<IndexType>())
        {
          return false;
        }
      }

      return true;
    });
}

mlir::comet::GpuTypeConverter::GpuTypeConverter(MLIRContext *context) 
: context(context){

    // std::cout << "Calling converter\n";
  addConversion([](Type type) { return type; });

  // Add encoding for tensor
  // addConversion([this](IndexType index) -> IntegerType {
  
  //   return mlir::IntegerType::get(this->context, 32);
  // });

  addConversion([this](MemRefType memrefType, SmallVectorImpl<Type> &results) -> LogicalResult {
    
    if(memrefType.getElementType().isa<IndexType>())
    {
      results.push_back(mlir::triton::PointerType::get( mlir::IntegerType::get(this->context, 32), 1));
    }
    else 
    {
      results.push_back(mlir::triton::PointerType::get(memrefType.getElementType(), 1));
    }
    for(size_t i = 0; i < 2; i++) // This is for strides of each dimension
    {
      results.push_back(mlir::IndexType::get(this->context));
    }
    
    for(size_t i = 0; i < 2; i++) // This is for sizes of each dimension
    {
      results.push_back(mlir::IndexType::get(this->context));
    }

    return success();
  });
  }

mlir::comet::GpuTypeConverter2::GpuTypeConverter2(MLIRContext *context) 
: context(context){

    // std::cout << "Calling converter\n";
  addConversion([](Type type) { return type; });

  // Add encoding for tensor
  // addConversion([this](IndexType index) -> IntegerType {
  
  //   return mlir::IntegerType::get(this->context, 32);
  // });

  // addConversion([this](MemRefType memrefType, SmallVectorImpl<Type> &results) -> LogicalResult {
    
  //   results.push_back(mlir::triton::PointerType::get(memrefType.getElementType(), 1));
  //   for(size_t i = 0; i < 2; i++) // This is for strides of each dimension
  //   {
  //     results.push_back(mlir::IndexType::get(this->context));
  //   }
    
  //   for(size_t i = 0; i < 2; i++) // This is for sizes of each dimension
  //   {
  //     results.push_back(mlir::IndexType::get(this->context));
  //   }

  //   return success();
  // });
  }