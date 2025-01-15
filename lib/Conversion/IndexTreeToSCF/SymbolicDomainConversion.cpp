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

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/Utils/Utils.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Conversion/IndexTreeToSCF/IndexTreeToSCF.h"
#include "comet/Dialect/IndexTree/Patterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/Transforms/DecomposeCallGraphTypes.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using llvm::SmallVector;

namespace mlir {
    namespace comet{
      #define GEN_PASS_DEF_CONVERTSYMBOLICDOMAINS
      #include "comet/Conversion/Passes.h.inc"
    }
}

struct SymbolicDomain {
  Value pos_size;
  Value pos_alloc_size;
  Value crd_size;
  Value dim_size;
  Value pos;
  Value mark_array;
};

static bool unpack_symbolic_domain(Value symbolic_domain, SymbolicDomain& result)
{
  if (auto cast = symbolic_domain.getDefiningOp<UnrealizedConversionCastOp>()) {
    result.pos_size = cast->getOperand(0);
    result.pos_alloc_size = cast->getOperand(1);
    result.crd_size = cast->getOperand(2);
    result.dim_size = cast->getOperand(3);
    result.pos = cast->getOperand(4);
    result.mark_array = cast->getOperand(5);
    return true;
  }
  return false;
}

namespace {
struct ConvertDomainInsertOp
    : public OpConversionPattern<indexTree::SymbolicDomainInsertOp> {
  using OpConversionPattern<indexTree::SymbolicDomainInsertOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(indexTree::SymbolicDomainInsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto context = op.getContext();
    Type index_type = rewriter.getIndexType();
    SymbolicDomain domain;
    if(!unpack_symbolic_domain(llvm::cast<indexTree::SymbolicDomainInsertOpAdaptor>(adaptor).getDomain(), domain)){
      return failure();
    }

    Value one = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1)); 
    if(op.getIsUnique())
    {
      // If we know the crd is unique, we can just increment the crd_size value
      domain.crd_size = rewriter.create<index::AddOp>(loc, index_type, domain.crd_size, one);
    } else 
    {
      
      Value mark = rewriter.create<index::AddOp>(loc, index_type, domain.pos_size, one);
      if(mark.getType() != domain.mark_array.getType().cast<ShapedType>().getElementType())
      {
        mark = rewriter.create<mlir::arith::IndexCastOp>(loc,  domain.mark_array.getType().cast<ShapedType>().getElementType() ,mark);
      }
      Value mark_val = rewriter.create<memref::LoadOp>(loc, domain.mark_array, op.getCrd());
      Value is_marked = rewriter.create<arith::CmpIOp>(loc, 
                                    rewriter.getI1Type(),
                                    arith::CmpIPredicate::eq, 
                                    mark,
                                    mark_val);
      scf::IfOp if_op = rewriter.create<scf::IfOp>(loc, index_type, is_marked, true);
      // We have seen this crd before
      rewriter.setInsertionPointToStart(if_op.thenBlock());
      rewriter.create<scf::YieldOp>(loc, domain.crd_size);

      // We haven't seen this crd before
      rewriter.setInsertionPointToStart(if_op.elseBlock());
      rewriter.create<memref::StoreOp>(loc, mark, domain.mark_array, op.getCrd());
      Value new_crd_size = rewriter.create<index::AddOp>(loc, index_type, domain.crd_size, one);
      rewriter.create<scf::YieldOp>(loc, new_crd_size);
      rewriter.setInsertionPointAfter(if_op);
      domain.crd_size = if_op.getResult(0);
    }

    Value materialized = getTypeConverter()->materializeArgumentConversion(
              rewriter, 
              op.getLoc(), 
              op.getDomain().getType(), 
              {
                domain.pos_size,
                domain.pos_alloc_size,
                domain.crd_size,
                domain.dim_size,
                domain.pos, 
                domain.mark_array
              }
    );
    rewriter.replaceOp(op, {materialized});
    return success();
  }
};

struct ConvertDomainEndRowOp
    : public OpConversionPattern<indexTree::SymbolicDomainEndRowOp> {
  using OpConversionPattern<indexTree::SymbolicDomainEndRowOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(indexTree::SymbolicDomainEndRowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Type index_type = rewriter.getIndexType();
    SymbolicDomain domain;
    if(!unpack_symbolic_domain(llvm::cast<indexTree::SymbolicDomainEndRowOpAdaptor>(adaptor).getDomain(), domain)){
      return failure();
    }

    Value inc = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
    Value new_pos_size = rewriter.create<index::AddOp>(loc, index_type, domain.pos_size, inc);
    Value crd_size_cast = rewriter.createOrFold<mlir::arith::IndexCastOp>(loc, rewriter.getIntegerType(op.getResult().getType().getIndicesBitwidth()), domain.crd_size);
    // TODO: Dynamically resize array?
    rewriter.create<memref::StoreOp>(loc, crd_size_cast, domain.pos, new_pos_size);
    Value materialized = getTypeConverter()->materializeArgumentConversion(
            rewriter,
            op.getLoc(),
            op.getDomain().getType(),
            {
              new_pos_size,
              domain.pos_alloc_size,
              domain.crd_size, 
              domain.dim_size,
              domain.pos, 
              domain.mark_array
            }
    );
    rewriter.replaceOp(op, {materialized});        
    return success();
  }
};

struct ConvertDomainDeclarationOp
    : public OpConversionPattern<indexTree::DeclDomainOp> {
  using OpConversionPattern<indexTree::DeclDomainOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(indexTree::DeclDomainOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Type indices_type = rewriter.getIntegerType(*op.getIndicesBitwidth());
    Type index_type = rewriter.getIndexType();
    Type memref_type = MemRefType::get({ShapedType::kDynamic,}, indices_type);

    Value zero = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
    Value inc = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1)); 
    Value pos_alloc_size = rewriter.create<index::AddOp>(loc, op.getNumRows(), inc);
    TypedValue<MemRefType> pos = rewriter.create<memref::AllocOp>(loc, memref_type, ValueRange{pos_alloc_size}, ValueRange(), nullptr);
    Value zero_cast = zero;
    zero_cast = rewriter.createOrFold<mlir::arith::IndexCastOp>(loc, pos.getType().getElementType(), zero);

    rewriter.create<memref::StoreOp>(loc, zero_cast, pos, zero);
    Value mark_array = rewriter.create<memref::AllocOp>(loc, memref_type, ValueRange{op.getDimSize()}, ValueRange(), nullptr);
    auto new_op = rewriter.create<UnrealizedConversionCastOp>(
      loc, 
      op->getResultTypes(),
      ValueRange({zero, pos_alloc_size, zero, op.getDimSize(), pos, mark_array})
    );
    rewriter.replaceOp(op, new_op->getResults());
    return success();
  }
};

struct ConvertSparseTensorOp
    : public OpConversionPattern<indexTree::IndexTreeSparseTensorOp> {
  using OpConversionPattern<indexTree::IndexTreeSparseTensorOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(indexTree::IndexTreeSparseTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    

    for(Value domain : llvm::cast<indexTree::IndexTreeSparseTensorOpAdaptor>(adaptor).getDomains())
    {
      if(llvm::isa<indexTree::SymbolicDomainType>(domain.getType())){
        if(!domain.getDefiningOp<UnrealizedConversionCastOp>())
          return failure();
      }
    }

    tensorAlgebra::SparseTensorType spType = mlir::cast<tensorAlgebra::SparseTensorType>(op->getResultTypes()[0]);
    auto ctx = op.getContext();
    auto format_unk = tensorAlgebra::TensorFormatEnumAttr::get(ctx, tensorAlgebra::TensorFormatEnum::UNK);
    auto format_dense = tensorAlgebra::TensorFormatEnumAttr::get(ctx, tensorAlgebra::TensorFormatEnum::D);
    auto format_compressed = tensorAlgebra::TensorFormatEnumAttr::get(ctx, tensorAlgebra::TensorFormatEnum::CU);
    SmallVector<Attribute, 3> dim_format;
    uint32_t rank = 0;

    auto loc = op.getLoc();
    Type index_type = rewriter.getIndexType();
    Type memref_type = MemRefType::get({ShapedType::kDynamic,}, spType.getIndicesType());
    Value zero = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
    Value one = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(1));
    Value nnz = one;
    llvm::SmallVector<Value, 3> dim_sizes, 
                                pos_indices, 
                                crd_indices, 
                                tile_pos_indices, 
                                tile_crd_indices;

    for(Value domain : llvm::cast<indexTree::IndexTreeSparseTensorOpAdaptor>(adaptor).getDomains())
    {
      rank += 1;
      if(llvm::isa<indexTree::DomainType>(domain.getType()))
      {
        Operation* domain_op = domain.getDefiningOp();
        if(llvm::isa<indexTree::IndexTreeDenseDomainOp>(domain_op))
        {
          auto dense_domain_op = llvm::cast<indexTree::IndexTreeDenseDomainOp>(domain_op);
          Value dim_size = dense_domain_op.getDimSize(); // Always index type
          Value dim_size_cast;
          dim_size_cast = rewriter.createOrFold<mlir::arith::IndexCastOp>(loc, spType.getIndicesType(), dim_size);


          Value pos = rewriter.create<memref::AllocOp>(loc,  MemRefType::get({ShapedType::kDynamic,}, spType.getIndicesType()), ValueRange({rewriter.create<index::ConstantOp>(loc, 1).getResult()}));
          rewriter.create<memref::StoreOp>(loc, dim_size_cast, pos, zero);
          Value crd = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, spType.getIndicesType()));
          Value pos_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, spType.getIndicesType()));
          Value crd_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, spType.getIndicesType()));

          pos_indices.push_back(rewriter.create<bufferization::ToTensorOp>(loc, pos, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
          crd_indices.push_back(rewriter.create<bufferization::ToTensorOp>(loc, crd, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
          tile_pos_indices.push_back(rewriter.create<bufferization::ToTensorOp>(loc, pos_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
          tile_crd_indices.push_back(rewriter.create<bufferization::ToTensorOp>(loc, crd_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));

          dim_sizes.push_back(dim_size);
          nnz = rewriter.create<index::MulOp>(loc, index_type, nnz, dim_size);
          dim_format.push_back(format_dense);
          dim_format.push_back(format_unk);
        } else if(llvm::isa<indexTree::IndexTreeSparseDomainOp>(domain_op))
        {
          auto sparse_domain_op = llvm::cast<indexTree::IndexTreeSparseDomainOp>(domain_op);
          Value dim_size = sparse_domain_op.getDimSize();
          Value crd_size = sparse_domain_op.getCrdSize();

          Value pos = sparse_domain_op.getPos();
          Value crd = sparse_domain_op.getCrd();
          Value pos_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, spType.getIndicesType()));
          Value crd_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, spType.getIndicesType()));

          pos_indices.push_back(pos);
          crd_indices.push_back(crd);

          tile_pos_indices.push_back(rewriter.create<bufferization::ToTensorOp>(loc, pos_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
          tile_crd_indices.push_back(rewriter.create<bufferization::ToTensorOp>(loc, crd_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));

          dim_sizes.push_back(dim_size);
          nnz = crd_size;
          dim_format.push_back(format_compressed);
          dim_format.push_back(format_unk);
        } else
          return failure();
      } else if(llvm::isa<indexTree::SymbolicDomainType>(domain.getType()))
      {
        SymbolicDomain domain_struct;
        assert(unpack_symbolic_domain(domain, domain_struct));
        Value crd = rewriter.create<memref::AllocOp>(loc, memref_type, ValueRange{domain_struct.crd_size}, ValueRange(), nullptr);
        Value pos_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, spType.getIndicesType()));
        Value crd_tile = rewriter.create<memref::AllocOp>(loc, MemRefType::get({0,}, spType.getIndicesType()));

        pos_indices.push_back(rewriter.create<bufferization::ToTensorOp>(loc, domain_struct.pos, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
        crd_indices.push_back(rewriter.create<bufferization::ToTensorOp>(loc, crd, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
        tile_pos_indices.push_back(rewriter.create<bufferization::ToTensorOp>(loc, pos_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));
        tile_crd_indices.push_back(rewriter.create<bufferization::ToTensorOp>(loc, crd_tile, rewriter.getUnitAttr(), rewriter.getUnitAttr()));

        dim_sizes.push_back(domain_struct.dim_size);
        nnz = domain_struct.crd_size;
        dim_format.push_back(format_compressed);
        dim_format.push_back(format_unk);
      }
    }

    //Allocate values array and initialize
    Type float_type = llvm::cast<tensorAlgebra::SparseTensorType>(op.getResult().getType()).getElementType();
    Value val_array = rewriter.create<memref::AllocOp>(loc, MemRefType::get({ShapedType::kDynamic,}, float_type), ValueRange{nnz}, ValueRange(), nullptr);
    Value float_zero = rewriter.create<arith::ConstantOp>(loc, float_type, rewriter.getFloatAttr(float_type, 0.0));
    auto for_loop = rewriter.create<scf::ForOp>(loc, zero, nnz, one);
    rewriter.setInsertionPointToStart(for_loop.getBody());
    auto induction_var = for_loop.getInductionVar();
    rewriter.create<memref::StoreOp>(loc, float_zero, val_array, induction_var);
    rewriter.setInsertionPointAfter(for_loop);
    val_array = rewriter.create<bufferization::ToTensorOp>(loc, val_array, rewriter.getUnitAttr(), rewriter.getUnitAttr());

    std::vector<Value> args;
    args.push_back(val_array);
    args.push_back(nnz);
    args.insert(args.end(), dim_sizes.begin(), dim_sizes.end());
    Value dims = rewriter.create<tensor::FromElementsOp>(loc, ValueRange(dim_sizes));

    rewriter.replaceOpWithNewOp<tensorAlgebra::SparseTensorConstructOp>(op, op.getResult().getType(), dims, pos_indices, crd_indices, tile_pos_indices, tile_crd_indices, val_array, rank, rewriter.getArrayAttr(dim_format));
    return success();
  }
};

class EraseDenseDomainOp : public OpConversionPattern<indexTree::IndexTreeDenseDomainOp>{
  public:
    using OpConversionPattern<indexTree::IndexTreeDenseDomainOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(indexTree::IndexTreeDenseDomainOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class EraseSparseDomainOp : public OpConversionPattern<indexTree::IndexTreeSparseDomainOp>{
  public:
    using OpConversionPattern<indexTree::IndexTreeSparseDomainOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(indexTree::IndexTreeSparseDomainOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} //namespace

struct ConvertSymbolicDomainsPass
    : public PassWrapper<ConvertSymbolicDomainsPass, OperationPass<func::FuncOp>>
{
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSymbolicDomainsPass)

  void runOnOperation() override
  {
    // Convert the rest of the index tree dialect to SCF
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion(
      [](indexTree::SymbolicDomainType domainType, SmallVectorImpl<Type> &types) {
        auto context = domainType.getContext();
        IntegerType indicesType = IntegerType::get(context, domainType.getIndicesBitwidth());
        Type index_type = IndexType::get(context);
        Type memref_type = MemRefType::get({ShapedType::kDynamic,}, indicesType);
        types.push_back(index_type);
        types.push_back(index_type);
        types.push_back(index_type);
        types.push_back(index_type);
        types.push_back(memref_type);
        types.push_back(memref_type);
        return success();
      });

    typeConverter.addSourceMaterialization(
      [](OpBuilder &builder, indexTree::SymbolicDomainType resultType, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        assert(inputs.size() == 6);
        Value value = builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)->getResult(0);
        return value;
      });

    typeConverter.addArgumentMaterialization(
      [](OpBuilder &builder, indexTree::SymbolicDomainType resultType, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        assert(inputs.size() == 6);
        Value value = builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)->getResult(0);
        return value;
      });

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<index::IndexDialect, arith::ArithDialect, scf::SCFDialect, memref::MemRefDialect, bufferization::BufferizationDialect, func::FuncDialect, tensor::TensorDialect>();
    target.addLegalDialect<tensorAlgebra::TADialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<indexTree::DeclDomainOp, indexTree::SymbolicDomainEndRowOp, indexTree::SymbolicDomainInsertOp>();
    target.addIllegalOp<indexTree::IndexTreeSparseTensorOp>();

    

    mlir::RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
    indexTree::populateIndexTreeTypeConversionPatterns(&getContext(), patterns, typeConverter, target);
    patterns.add<ConvertDomainDeclarationOp, ConvertDomainInsertOp, ConvertDomainEndRowOp, ConvertSparseTensorOp>(typeConverter, &getContext());
    patterns.add<EraseDenseDomainOp, EraseSparseDomainOp>(typeConverter, &getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

/// Lower sparse tensor algebra operation to loops
std::unique_ptr<Pass> mlir::comet::createConvertSymbolicDomainsPass()
{
  return std::make_unique<ConvertSymbolicDomainsPass>();
}