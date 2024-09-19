
#include <list>
#include <memory>
#include "comet/Conversion/GpuToTriton/GpuToTritonPass.h"
#include "comet/Conversion/GpuToTriton/GpuToTritonConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Casting.h"

#include <map>
#include <set>

#define GEN_PASS_CLASSES
#include "comet/Conversion/GpuToTriton/Passes.h.inc"
using namespace mlir;

namespace {

mlir::Value getSymOrDimOperand(affine::AffineApplyOp affineOp, AffineExpr exp)
{
  if(exp.getKind() == mlir::AffineExprKind::DimId)
  {
    auto dim = llvm::cast<mlir::AffineDimExpr>(exp);
    return affineOp.getDimOperands()[dim.getPosition()];
  }
  else if(exp.getKind() == mlir::AffineExprKind::SymbolId)
  {
    auto dim = llvm::cast<mlir::AffineSymbolExpr>(exp);
    return affineOp.getSymbolOperands()[dim.getPosition()];
  }
  else 
  {
    return mlir::Value();
  }
}

affine::AffineApplyOp create_simplified_affine_apply_op(affine::AffineApplyOp op, ConversionPatternRewriter &rewriter)
{
    auto affineMap = op.getAffineMap();
    llvm::SmallVector<mlir::Value> operands;
    auto newMap = mlir::AffineMap::get( affineMap.getNumDims(),  affineMap.getNumSymbols(),affineMap.getResult(0));
    for(auto operand:op.getMapOperands())
    {
      operands.push_back(operand);
    }

    mlir::affine::fullyComposeAffineMapAndOperands(&newMap, &operands);
    auto newAffine = rewriter.create<affine::AffineApplyOp>(op->getLoc(), newMap, operands);
    op->replaceAllUsesWith(newAffine);
    for(auto v: op.getMapOperands())
    {
      // COMET_ERRS << v;
      if(std::find(operands.begin(), operands.end(), v) == std::end(operands))
      {
        if(mlir::isa_and_nonnull<affine::AffineApplyOp>(v.getDefiningOp()) )
        {

          // COMET_ERRS << "FOUND\n";
          if(v.hasOneUse())
          {
            // COMET_ERRS << "One use\n";
            rewriter.eraseOp(v.getDefiningOp());
          }
        }

      }
    }


    rewriter.eraseOp(op);

    return newAffine;
}

void makeShapesEqual(Operation* op, mlir::Value& lhs, mlir::Value& rhs, ConversionPatternRewriter& rewriter)
{
auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if(lhsT != rhsT)
  {
    if(auto lhsTensor = lhsT.dyn_cast<RankedTensorType>())
    {
      if(auto rhsTensor = rhsT.dyn_cast<RankedTensorType>() ) // 
      {

        if(lhsTensor.getRank() != rhsTensor.getRank())
        {
          if(lhsTensor.getRank()>rhsTensor.getRank())
          {
            if(lhsTensor.getDimSize(0) > 1)
            {
              lhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), RankedTensorType::get({lhsTensor.getDimSize(0), rhsTensor.getDimSize(0)}  , lhsTensor.getElementType()), lhs);
              rhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), lhs.getType(), rewriter.create<triton::ExpandDimsOp>(op->getLoc(), rhs, 0)); 
            }
            else 
            {
              lhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), RankedTensorType::get({rhsTensor.getDimSize(0), lhsTensor.getDimSize(1)}  , lhsTensor.getElementType()), lhs);
              rhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), lhs.getType(), rewriter.create<triton::ExpandDimsOp>(op->getLoc(), rhs, 1)); 
            }
          }
          else 
          {
            if(rhsTensor.getDimSize(0) > 1)
            {
              rhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), RankedTensorType::get({rhsTensor.getDimSize(0), lhsTensor.getDimSize(0)}  , rhsTensor.getElementType()), rhs);
              lhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), rhs.getType(), rewriter.create<triton::ExpandDimsOp>(op->getLoc(), lhs, 0)); 
            }
            else 
            {
              rhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), RankedTensorType::get({lhsTensor.getDimSize(0), rhsTensor.getDimSize(1)}  , rhsTensor.getElementType()), rhs);
              lhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), rhs.getType(), rewriter.create<triton::ExpandDimsOp>(op->getLoc(), lhs, 1)); 
            }
          }
        }

        else
        {
          if (lhsTensor.getDimSize(0) == 1 && rhsTensor.getDimSize(0) > 1)
          {
            lhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), RankedTensorType::get({rhsTensor.getDimSize(0), lhsTensor.getDimSize(1)}  , lhsTensor.getElementType()), lhs);
          }
          
          if (rhsTensor.getDimSize(0) == 1 && lhsTensor.getDimSize(0) > 1)
          {
            rhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), RankedTensorType::get({lhsTensor.getDimSize(0), rhsTensor.getDimSize(1)}  , rhsTensor.getElementType()), rhs);
          }
          
          if (lhsTensor.getDimSize(1) == 1 && rhsTensor.getDimSize(1) > 1)
          {
            lhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), RankedTensorType::get({lhsTensor.getDimSize(0), rhsTensor.getDimSize(1)}  , lhsTensor.getElementType()), lhs);
          }
          if (rhsTensor.getDimSize(1) == 1 && lhsTensor.getDimSize(1) > 1)
          {
            rhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), RankedTensorType::get({rhsTensor.getDimSize(0), lhsTensor.getDimSize(1)}  , rhsTensor.getElementType()), rhs);
          }
        }
        // for(int i = 0; i < lhsTensor.getRank(); i++)
        // {
        //   if(lhsTensor.getDimSize(i) != rhsTensor.getDimSize(i))
        //   {
        //     if(lhsTensor.getDimSize(i) == 1)
        //     {
        //       lhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), RankedTensorType::get(rhsTensor.getShape()[i], rewriter.getI32Type()), lhs);
        //     }
        //     else if (rhsTensor.getDimSize(i) == 1)
        //     {
        //       rhs = rewriter.create<triton::BroadcastOp>(op->getLoc(), RankedTensorType::get(lhsTensor.getShape(), rewriter.getI32Type()), rhs);
        //     }
        //   }

        // }
      }
      else 
      {
        rhs = rewriter.createOrFold<triton::SplatOp>(op->getLoc(), RankedTensorType::get(lhsTensor.getShape(), rhs.getType()), rhs);
      }
    }
    else
    {
      if(auto rhsTensor = rhsT.dyn_cast<RankedTensorType>())
      {
        lhs = rewriter.createOrFold<triton::SplatOp>(op->getLoc(), RankedTensorType::get(rhsTensor.getShape(), lhs.getType()), lhs);
      }
    } 
  }
}

void handleBinaryExpr(Operation* op, std::map<const void*, mlir::Value>& m, AffineExpr& exp, ConversionPatternRewriter& rewriter)
{
  auto binOp = llvm::cast<mlir::AffineBinaryOpExpr>(exp);
  auto lhs = m[binOp.getLHS().getAsOpaquePointer()];
  auto rhs = m[binOp.getRHS().getAsOpaquePointer()];
  // auto lhsT = lhs.getType();
  // auto rhsT = rhs.getType();

  makeShapesEqual(op, lhs, rhs, rewriter);

  switch (exp.getKind()) {
    case  AffineExprKind::Mul :{
    if(m.find(exp.getAsOpaquePointer()) == m.end())
        m[exp.getAsOpaquePointer()] = rewriter.createOrFold<arith::MulIOp>(op->getLoc(), lhs, rhs);
    } break;
    case  AffineExprKind::Add :{
      if(m.find(exp.getAsOpaquePointer()) == m.end())
        m[exp.getAsOpaquePointer()] = rewriter.create<arith::AddIOp>(op->getLoc(), lhs, rhs); 
    } break;
    case AffineExprKind::Mod:
    case AffineExprKind::FloorDiv:
    case AffineExprKind::CeilDiv:
    case AffineExprKind::Constant:
    case AffineExprKind::DimId:
    case AffineExprKind::SymbolId:
      break;
    }

  return;
}

LogicalResult convertMemoryOp(Operation* op, ConversionPatternRewriter &rewriter) 
{
  int mem_offset;
  int index_offset;
  if (mlir::isa<memref::LoadOp>(op))
  {
    mem_offset = 0;
    index_offset = 1;
  }
  else if (mlir::isa<memref::StoreOp>(op))
  {
    mem_offset = 1;
    index_offset = 2;
  }
  else 
  {
    return failure();
  }

  std::vector<Value> block_strides;
  std::vector<int> dims;
  std::vector<Value> global_block_ids;
  std::vector<int64_t> block_sizes;
  std::vector<Value> tensor_sizes;
  mlir::Value memMask;
  for(size_t i = index_offset; i < op->getNumOperands(); i++)
  {
    auto affineOp = llvm::dyn_cast_if_present<affine::AffineApplyOp>(op->getOperand(i).getDefiningOp());
    auto minOp = llvm::dyn_cast_if_present<arith::MinUIOp>(op->getOperand(i).getDefiningOp());
    if(affineOp || (minOp && (minOp->hasAttr("GuardX") || minOp->hasAttr("GuardY") || minOp->hasAttr("GuardR"))))
    {
      // int pidXIndex = -1;
      // int pidYIndex = -1;
      // int dim = -1;
      Value guardX = NULL, guardY = NULL, guardR = NULL;
      Value guardXExpr = NULL, guardYExpr = NULL, guardRExpr = NULL;
      std::map<const void*, mlir::Value> map, mapGuard;
      mlir::Value bidX, bidY, tidX, tidY;
      if (affineOp)
      {
        for(auto& oper: affineOp->getOpOperands())
        {
          if(auto defOp = llvm::dyn_cast_or_null<arith::MinUIOp>(oper.get().getDefiningOp()))
          {
            if(defOp->hasAttr("GuardX"))
            {
              guardX = defOp.getRhs();
              guardXExpr = defOp.getLhs();
              oper.set(defOp.getLhs());
            }
            else if (defOp->hasAttr("GuardY"))
            {
              guardY = defOp.getRhs();
              guardYExpr = defOp.getLhs();

              oper.set(defOp.getLhs());
            }
            else if (defOp->hasAttr("GuardR"))
            {
              guardR = defOp.getRhs();
              guardRExpr = defOp.getLhs();

              oper.set(defOp.getLhs());
            }
          }
        }
      }
      else if(minOp)
      {
        if(minOp->hasAttr("GuardX"))
        {
          guardX = minOp.getRhs();
          guardXExpr = minOp.getLhs();
        }
        else if (minOp->hasAttr("GuardY"))
        {
          guardY = minOp.getRhs();
          guardYExpr = minOp.getLhs();
        }
        else if (minOp->hasAttr("GuardR"))
        {
          guardR = minOp.getRhs();
          guardRExpr = minOp.getLhs();
        }
        affineOp = cast<affine::AffineApplyOp>(minOp.getLhs().getDefiningOp());
      }
      else 
      {
        return mlir::failure();
      }
      
      auto otherFunc = [&map, &mapGuard, &op, &rewriter, &bidX, &bidY, &tidX, &tidY, &guardX, &guardY, &guardR](affine::AffineApplyOp& aaffineop)  {
      return [&aaffineop, &map, &mapGuard, &op, &rewriter, &bidX, &bidY, &tidX, &tidY, &guardX, &guardY, &guardR](AffineExpr exp) -> void{
        if(exp.getKind() == mlir::AffineExprKind::DimId || exp.getKind() == mlir::AffineExprKind::SymbolId)
        {
          if(auto bId = mlir::dyn_cast_or_null<mlir::gpu::BlockIdOp>(getSymOrDimOperand(aaffineop, exp).getDefiningOp()))
          {
            if(bId.getDimension() == mlir::gpu::Dimension::x)
            {
              if(map.find(exp.getAsOpaquePointer()) == map.end())
              {
                map[exp.getAsOpaquePointer()] = rewriter.create<triton::GetProgramIdOp>(op->getLoc(), rewriter.getI32Type(), mlir::triton::ProgramIDDimAttr::get(op->getContext(), mlir::triton::ProgramIDDim::X));
                if(guardX)
                {
                  mapGuard[exp.getAsOpaquePointer()] = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 0, 32);
                }
              }
            
              bidX = map[exp.getAsOpaquePointer()];
            }
            else if(bId.getDimension() == mlir::gpu::Dimension::y)
            {
              if(map.find(exp.getAsOpaquePointer()) == map.end())
              {
                map[exp.getAsOpaquePointer()] = rewriter.create<triton::GetProgramIdOp>(op->getLoc(), rewriter.getI32Type(), mlir::triton::ProgramIDDimAttr::get(op->getContext(), mlir::triton::ProgramIDDim::Y));
                if(guardY)
                {
                  mapGuard[exp.getAsOpaquePointer()] = rewriter.create<arith::ConstantIntOp>(op->getLoc(), 0, 32);
                }
              }
              
              bidY = map[exp.getAsOpaquePointer()];
            }
          }
          else if(auto tId = mlir::dyn_cast_or_null<mlir::gpu::ThreadIdOp>(getSymOrDimOperand(aaffineop, exp).getDefiningOp()))
          {
            if(tId.getDimension() == mlir::gpu::Dimension::x)
            {
              if(map.find(exp.getAsOpaquePointer()) == map.end())
              {
                int blockX = op->getParentOfType<triton::FuncOp>()->getAttrOfType<IntegerAttr>("block_size_x").getInt();
                auto range = rewriter.create<mlir::triton::MakeRangeOp>(op->getLoc(), RankedTensorType::get({blockX} , rewriter.getI32Type()), 0, blockX)->getResult(0);
                map[exp.getAsOpaquePointer()] = rewriter.create<triton::ExpandDimsOp>(op->getLoc(), RankedTensorType::get({1, blockX}, rewriter.getI32Type()), range, 0); 
                if(guardX)
                {
                  mapGuard[exp.getAsOpaquePointer()] = rewriter.create<triton::SplatOp>(op->getLoc(), RankedTensorType::get({1, blockX}, rewriter.getI32Type()), guardX );
                }
              }
              tidX = map[exp.getAsOpaquePointer()];
            }
            else if(tId.getDimension() == mlir::gpu::Dimension::y)
            {
              if(map.find(exp.getAsOpaquePointer()) == map.end())
              {
                int blockY = op->getParentOfType<triton::FuncOp>()->getAttrOfType<IntegerAttr>("block_size_y").getInt();
                auto range = rewriter.create<mlir::triton::MakeRangeOp>(op->getLoc(), RankedTensorType::get({blockY} , rewriter.getI32Type()), 0, blockY)->getResult(0);
                map[exp.getAsOpaquePointer()] = rewriter.create<triton::ExpandDimsOp>(op->getLoc(), RankedTensorType::get({blockY, 1}, rewriter.getI32Type()), range, 1); 
                if(guardY)
                {
                  mapGuard[exp.getAsOpaquePointer()] = rewriter.create<triton::SplatOp>(op->getLoc(), RankedTensorType::get({blockY, 1}, rewriter.getI32Type()), guardY );
                }

              }
              tidY = map[exp.getAsOpaquePointer()];

            }
          }
          // [TODO] else if(auto redIdx = mlir::dyn_cast_or_null<IntToRedIndexOp>(getSymOrDimOperand(aaffineop, exp).getDefiningOp()))
          else if(getSymOrDimOperand(aaffineop, exp).getDefiningOp() != NULL && getSymOrDimOperand(aaffineop, exp).getDefiningOp()->hasAttr("ReductionIndex"))
          {
            // assert(false);
            auto redIdx = getSymOrDimOperand(aaffineop, exp).getDefiningOp()->getResult(0);
            if(map.find(exp.getAsOpaquePointer()) == map.end())
            {
              // [TODO] if(auto blockArg = mlir::dyn_cast_or_null<BlockArgument>(redIdx.getIn().getDefiningOp()->getOperand(0)))
              if(auto blockArg = mlir::dyn_cast_or_null<BlockArgument>(redIdx.getDefiningOp()->getOperand(0)))
              {
                auto loopBlockSize = blockArg.getOwner()->getParentOp()->getAttrOfType<IntegerAttr>("loop_block_size").getInt();
                map[exp.getAsOpaquePointer()] = rewriter.createOrFold<triton::MakeRangeOp>(op->getLoc(), RankedTensorType::get({loopBlockSize}, rewriter.getI32Type()),  0, loopBlockSize);
                if(guardR)
                {
                  mapGuard[exp.getAsOpaquePointer()] = rewriter.create<triton::SplatOp>(op->getLoc(), RankedTensorType::get({loopBlockSize}, rewriter.getI32Type()), guardR );
                }

              }
              else 
              {
                // [TODO] map[exp.getAsOpaquePointer()] = redIdx.getIn();
                map[exp.getAsOpaquePointer()] = redIdx;
                if(guardR)
                {
                  mapGuard[exp.getAsOpaquePointer()] = guardR;
                }
              }
            }
          }
          else 
          {
            if(map.find(exp.getAsOpaquePointer()) == map.end())
            {
              // COMET_ERRS << "HERE\n";
              if(getSymOrDimOperand(aaffineop, exp).getType().isa<IndexType>())
              {
                // COMET_ERRS << "HERE\n";
                // COMET_ERRS << getSymOrDimOperand(aaffineop, exp);
                map[exp.getAsOpaquePointer()] = rewriter.create<arith::IndexCastOp>(op->getLoc(), rewriter.getI32Type(), getSymOrDimOperand(aaffineop, exp))->getResult(0);
                if(auto barg = dyn_cast_or_null<BlockArgument>(getSymOrDimOperand(aaffineop, exp)))
                {
                  if(!(barg.getOwner()->getParentOp()->hasAttr("programs_loop_x") || barg.getOwner()->getParentOp()->hasAttr("programs_loop_y")|| barg.getOwner()->getParentOp()->hasAttr("loop_block_size")))
                  {
                    if(guardR || guardX || guardY)
                    {
                      mapGuard[exp.getAsOpaquePointer()] = rewriter.create<arith::IndexCastOp>(op->getLoc(), rewriter.getI32Type(), getSymOrDimOperand(aaffineop, exp))->getResult(0);
                    }
                  }
                  else 
                  {
                    if(guardR || guardX || guardY)
                    {
                      mapGuard[exp.getAsOpaquePointer()] = rewriter.create<arith::ConstantIntOp>(aaffineop->getLoc(), 0, 32);
                    }
                  }
                }
                else 
                {
                  if(guardR || guardX || guardY)
                  {
                    mapGuard[exp.getAsOpaquePointer()] = rewriter.create<arith::IndexCastOp>(op->getLoc(), rewriter.getI32Type(), getSymOrDimOperand(aaffineop, exp))->getResult(0);
                  }
                }
              }
              else 
              {
                map[exp.getAsOpaquePointer()] = getSymOrDimOperand(aaffineop, exp);
                if(auto barg = dyn_cast_or_null<BlockArgument>(getSymOrDimOperand(aaffineop, exp)))
                {
                  if(!(barg.getOwner()->getParentOp()->hasAttr("programs_loop_x") || barg.getOwner()->getParentOp()->hasAttr("programs_loop_y" ) || barg.getOwner()->getParentOp()->hasAttr("loop_block_size")))
                  {
                    if(guardR || guardX || guardY)
                    {
                      mapGuard[exp.getAsOpaquePointer()] = getSymOrDimOperand(aaffineop, exp);
                    }
                  }
                  else 
                  {
                    if(guardR || guardX || guardY)
                    {
                      mapGuard[exp.getAsOpaquePointer()] = rewriter.create<arith::ConstantIntOp>(aaffineop->getLoc(), 0, 32);
                    }
                  }
                }
                else 
                {
                  if(guardR || guardX || guardY)
                  {
                    mapGuard[exp.getAsOpaquePointer()] = getSymOrDimOperand(aaffineop, exp);
                  }
                }

              }
            }
          }
        }
        else if(exp.getKind() == mlir::AffineExprKind::Constant)
        {

          if(map.find(exp.getAsOpaquePointer()) == map.end())
          {
            auto cst = llvm::cast<mlir::AffineConstantExpr>(exp);
            
            map[exp.getAsOpaquePointer()] = rewriter.create<arith::ConstantIntOp>(aaffineop->getLoc(), cst.getValue(), 32);
            if(guardR || guardX || guardY)
            {
              mapGuard[exp.getAsOpaquePointer()] = rewriter.create<arith::ConstantIntOp>(aaffineop->getLoc(), cst.getValue(), 32);
            }
          }
        }
        else
        {
          // auto binOp = llvm::cast<mlir::AffineBinaryOpExpr>(exp);
          // for(auto m: {map /*mapGuard*/})
          handleBinaryExpr(op, map, exp, rewriter);
          if(!mapGuard.empty())
          {
            handleBinaryExpr(op, mapGuard, exp, rewriter);
          }
        }
      };};

      Value resultGuard = NULL;
      if(guardXExpr)
      {
        map.clear();
        mapGuard.clear();
        auto op = cast<affine::AffineApplyOp>(guardXExpr.getDefiningOp());
        cast<affine::AffineApplyOp>(guardXExpr.getDefiningOp()).getAffineMap().getResult(0).walk(otherFunc(op));
        // llvm::errs() << " GuardX :" << cast<affine::AffineApplyOp>(guardXExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer() << "\n";
        // guardXExpr.dump();
        // map[cast<affine::AffineApplyOp>(guardXExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer()].dump();
        guardX = map[cast<affine::AffineApplyOp>(guardXExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer()];
        guardXExpr.replaceUsesWithIf(guardX, [](OpOperand& oper){return !isa<arith::MinUIOp>(oper.getOwner());});
        
        resultGuard = rewriter.create<arith::CmpIOp>(op->getLoc(), arith::CmpIPredicate::slt, map[cast<affine::AffineApplyOp>(guardXExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer()], mapGuard[cast<affine::AffineApplyOp>(guardXExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer()]);
      }
      if(guardYExpr)
      {
        map.clear();
        mapGuard.clear();
        auto op = cast<affine::AffineApplyOp>(guardYExpr.getDefiningOp());
        cast<affine::AffineApplyOp>(guardYExpr.getDefiningOp()).getAffineMap().getResult(0).walk(otherFunc(op));
        auto myGuard = rewriter.create<arith::CmpIOp>(op->getLoc(), arith::CmpIPredicate::slt, map[cast<affine::AffineApplyOp>(guardYExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer()], mapGuard[cast<affine::AffineApplyOp>(guardYExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer()]).getResult();
        if(!resultGuard) {
          resultGuard = myGuard;
        }
        else 
        {
          auto lhs = resultGuard;
          auto rhs = myGuard;
          makeShapesEqual(op, lhs, rhs, rewriter);
          resultGuard = rewriter.create<arith::AndIOp>(op->getLoc(), lhs, rhs);
        }

        // llvm::errs() << " GuardY :" << cast<affine::AffineApplyOp>(guardYExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer() << "\n";
        // guardYExpr.dump();

        // map[cast<affine::AffineApplyOp>(guardYExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer()].dump();
        guardY = map[cast<affine::AffineApplyOp>(guardYExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer()];
        guardYExpr.replaceUsesWithIf(guardY, [](OpOperand& oper){return !isa<arith::MinUIOp>(oper.getOwner());});

      }
      if(guardRExpr)
      {
        map.clear();
        mapGuard.clear();
        auto op = cast<affine::AffineApplyOp>(guardRExpr.getDefiningOp());
        cast<affine::AffineApplyOp>(guardRExpr.getDefiningOp()).getAffineMap().getResult(0).walk(otherFunc(op));
        auto myGuard = rewriter.create<arith::CmpIOp>(op->getLoc(), arith::CmpIPredicate::slt, map[cast<affine::AffineApplyOp>(guardRExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer()], mapGuard[cast<affine::AffineApplyOp>(guardRExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer()]).getResult();
        if(!resultGuard) 
        {
          resultGuard = myGuard;
        }
        else 
        {
          auto lhs = resultGuard;
          auto rhs = myGuard;
          makeShapesEqual(op, lhs, rhs, rewriter);
          resultGuard = rewriter.create<arith::AndIOp>(op->getLoc(), lhs, rhs);
        }

        // llvm::errs() << " GuardR :" << cast<affine::AffineApplyOp>(guardRExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer() <<  "\n";
        // guardRExpr.dump();

        // map[cast<affine::AffineApplyOp>(guardRExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer()].dump();
        guardR =  map[cast<affine::AffineApplyOp>(guardRExpr.getDefiningOp()).getAffineMap().getResult(0).getAsOpaquePointer()];
        guardRExpr.replaceUsesWithIf(guardR, [](OpOperand& oper){return !isa<arith::MinUIOp>(oper.getOwner());});
      }

      map.clear();
      mapGuard.clear();
      affineOp.getAffineMap().getResult(0).walk(otherFunc(affineOp));

      // op->getParentOfType<ModuleOp>()->dump();
      // COMET_ERRS << map[affineOp.getAffineMap().getResult(0).getAsOpaquePointer()];
      mlir::Value res = map[affineOp.getAffineMap().getResult(0).getAsOpaquePointer()];
      // res.dump();
      // res.getParentBlock()->dump();
      // affineOp->replaceAllUsesWith(res.getDefiningOp());
      // rewriter.eraseOp(affineOp);
      op->setOperand(i, res);
      memMask = resultGuard;
    }
    else if (isa<scf::ForOp>(op->getParentOp()) && isa<BlockArgument>(op->getOperand(i)))
    {
      auto res = rewriter.create<arith::IndexCastOp>(op->getLoc(), rewriter.getI32Type(), op->getOperand(i));
      rewriter.replaceAllUsesExcept(op->getOperand(i), res, res);
    }
  }

  mlir::Value ptr;
  if(RankedTensorType t = op->getOperand(mem_offset).getType().dyn_cast<RankedTensorType>())  
  {
    if(t.getElementType().isa<IndexType>()) 
    {
      ptr = rewriter.create<arith::IndexCastOp>(op->getLoc(), RankedTensorType::get(t.getShape(), rewriter.getI32Type()), op->getOperand(mem_offset));
    }
  }
  else if(op->getOperand(mem_offset).getType().isa<IndexType>())
  {
    ptr = rewriter.create<arith::IndexCastOp>(op->getLoc(), rewriter.getI32Type(), op->getOperand(mem_offset));
  } 
  else 
  {
    ptr = op->getOperand(mem_offset);
  }
  // auto ptr_array = rewriter.create<triton::SplatOp>(op->getLoc(), RankedTensorType::get(block_sizes, ptr.getType()), ptr);
  mlir::Value ptr_array;
  if(op->getOperand(mem_offset +1).getType().isa<RankedTensorType>())
  {
    ptr_array = rewriter.create<triton::SplatOp>(op->getLoc(), RankedTensorType::get(op->getOperand(mem_offset +1).getType().cast<RankedTensorType>().getShape(), ptr.getType()), ptr).getResult();
  }
  else 
  {
    ptr_array = ptr;
  }

  auto final_ptr_array = rewriter.create<triton::AddPtrOp>(op->getLoc(), ptr_array.getType(), ptr_array, op->getOperand(mem_offset +1));

  if (mlir::isa<memref::LoadOp>(op))
  {
    mlir::triton::LoadOp loadVal;
    if(memMask)
    {
      loadVal = rewriter.create<triton::LoadOp>(op->getLoc(), final_ptr_array,  memMask, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    }
    else
    {
      loadVal = rewriter.create<triton::LoadOp>(op->getLoc(), final_ptr_array, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    }

    rewriter.replaceAllUsesWith(op->getResult(0), loadVal);
    rewriter.eraseOp(op);
  }
  else 
  {
    mlir::Value toStore;
    if (op->getOperand(0).getType().dyn_cast<RankedTensorType>() ) {
      toStore = op->getOperand(0);
    }
    else if(auto ptr_array_type = final_ptr_array.getResult().getType().dyn_cast<RankedTensorType>(); ptr_array_type && !op->getOperand(0).getType().dyn_cast<RankedTensorType>()  ) {
      toStore = rewriter.create<triton::SplatOp>(op->getLoc(), RankedTensorType::get(ptr_array_type.getShape(), op->getOperand(0).getType()), op->getOperand(0));
    }
    mlir::triton::StoreOp storeVal;
    if(memMask)
    {
      storeVal = rewriter.create<triton::StoreOp>(op->getLoc(), final_ptr_array, toStore, memMask, mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
    }
    else 
    {
      storeVal = rewriter.create<triton::StoreOp>(op->getLoc(), final_ptr_array, toStore,  mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
    }
    rewriter.replaceOp(op, storeVal);
  }

  return success();
}

template<typename T>
LogicalResult ExpandScalarTensorArithOp(T op, ConversionPatternRewriter &rewriter)
{
  mlir::Value lhs = op.getOperand(0);
  mlir::Value rhs = op.getOperand(1);
  mlir::Value res = op.getResult();
  mlir::Type lhs_type = lhs.getType();
  mlir::Type rhs_type = rhs.getType();  
  mlir::Type res_type = res.getType();  

  Operation* expandedOp = NULL;
  if(lhs_type != rhs_type)
  {
    if(auto lhs_tensor_type = lhs_type.dyn_cast_or_null<RankedTensorType>())
    {
      if(!rhs_type.isa<RankedTensorType>()  && lhs_tensor_type.getElementType() == rhs_type)
      {
        auto scalarExpanded = rewriter.create<triton::SplatOp>(op->getLoc(), lhs.getType(), rhs);
        expandedOp = rewriter.create<T>(op->getLoc(), lhs, scalarExpanded);
      }
    }
    else if(auto rhs_tensor_type = rhs_type.dyn_cast_or_null<RankedTensorType>())
    {
      if(rhs_tensor_type.getElementType() == lhs_type)
      {
        auto scalarExpanded = rewriter.create<triton::SplatOp>(op->getLoc(), rhs.getType(), lhs);
        expandedOp = rewriter.create<T>(op->getLoc(), scalarExpanded, rhs);
      }          
    }
  }
  else if (lhs_type != res_type)
  {
    expandedOp = rewriter.create<T>(op->getLoc(), op->getOperand(0), op->getOperand(1));
  }

  if(expandedOp)
  {
    // rewriter.replaceAllUsesWith(op, expandedOp);
    op->replaceAllUsesWith(expandedOp);

    for(auto& u: expandedOp->getUses())
    {
      if(dyn_cast<triton::SplatOp>(u.getOwner()))
      {
        rewriter.replaceAllUsesWith(u.getOwner()->getResult(0), expandedOp->getResult(0));
        rewriter.eraseOp(u.getOwner());
      }
    }
    rewriter.eraseOp(op);

    return success();
  }

  return failure();
}
LogicalResult ExpandScalarTensorSelectOp(arith::SelectOp& op, ConversionPatternRewriter &rewriter)
{
  mlir::Value cond = op.getOperand(0);
  mlir::Value lhs = op.getOperand(1);
  mlir::Value rhs = op.getOperand(2);
  mlir::Value res = op.getResult();
  mlir::Type cond_type = cond.getType();
  mlir::Type lhs_type = lhs.getType();
  mlir::Type rhs_type = rhs.getType();  
  mlir::Type res_type = res.getType();  

  Operation* expandedOp = NULL;
  if(lhs_type != rhs_type)
  {
    if(auto lhs_tensor_type = lhs_type.dyn_cast_or_null<RankedTensorType>())
    {
      if(!cond_type.isa<RankedTensorType>())
      {
        auto expandedCond = rewriter.create<triton::SplatOp>(op->getLoc(), RankedTensorType::get(lhs_tensor_type.getShape(), cond.getType()), cond);
        cond = expandedCond.getResult();
      }
      if(!rhs_type.isa<RankedTensorType>()  && lhs_tensor_type.getElementType() == rhs_type)
      {
        auto scalarExpanded = rewriter.create<triton::SplatOp>(op->getLoc(), lhs.getType(), rhs);
        expandedOp = rewriter.create<arith::SelectOp>(op->getLoc(), cond, lhs, scalarExpanded);
      }
    }
    else if(auto rhs_tensor_type = rhs_type.dyn_cast_or_null<RankedTensorType>())
    {
      if(!cond_type.isa<RankedTensorType>())
      {
        auto expandedCond = rewriter.create<triton::SplatOp>(op->getLoc(), RankedTensorType::get(rhs_tensor_type.getShape(), cond.getType()), cond);
        cond = expandedCond.getResult();
      }
      if(rhs_tensor_type.getElementType() == lhs_type)
      {
        auto scalarExpanded = rewriter.create<triton::SplatOp>(op->getLoc(), rhs.getType(), lhs);
        expandedOp = rewriter.create<arith::SelectOp>(op->getLoc(), cond, scalarExpanded, rhs);
      }          
    }
  }
  else if (lhs_type != res_type)
  {
    auto rhs_tensor_type = rhs_type.dyn_cast_or_null<RankedTensorType>();
    if(rhs_tensor_type && !cond_type.isa<RankedTensorType>())
    {
      auto expandedCond = rewriter.create<triton::SplatOp>(op->getLoc(), RankedTensorType::get(rhs_tensor_type.getShape(), cond.getType()), cond);
      cond = expandedCond.getResult();
    }
    expandedOp = rewriter.create<arith::SelectOp>(op->getLoc(), cond, lhs, rhs);
  }
  else if (!cond_type.isa<RankedTensorType>() && rhs_type.isa<RankedTensorType>())
  {
    auto rhs_tensor_type = rhs_type.cast<RankedTensorType>();
    auto expandedCond = rewriter.create<triton::SplatOp>(op->getLoc(), RankedTensorType::get(rhs_tensor_type.getShape(), cond.getType()), cond);
    cond = expandedCond.getResult();
    expandedOp = rewriter.create<arith::SelectOp>(op->getLoc(), cond, lhs, rhs);
  }


  if(expandedOp)
  {
    // rewriter.replaceAllUsesWith(op, expandedOp);
    op->replaceAllUsesWith(expandedOp);

    for(auto& u: expandedOp->getUses())
    {
      if(dyn_cast<triton::SplatOp>(u.getOwner()))
      {
        rewriter.replaceAllUsesWith(u.getOwner()->getResult(0), expandedOp->getResult(0));
        rewriter.eraseOp(u.getOwner());
      }
    }
    rewriter.eraseOp(op);

    return success();
  }

  return failure();
}

template<typename T, typename TAttr>
LogicalResult ExpandScalarTensorArithCmpOp(T op, ConversionPatternRewriter &rewriter)
{
  mlir::Value lhs = op.getOperand(0);
  mlir::Value rhs = op.getOperand(1);
  mlir::Value res = op.getResult();
  mlir::Type lhs_type = lhs.getType();
  mlir::Type rhs_type = rhs.getType();  
  mlir::Type res_type = res.getType();  

  Operation* expandedOp = NULL;
  if(lhs_type != rhs_type)
  {
    if(auto lhs_tensor_type = lhs_type.dyn_cast_or_null<RankedTensorType>())
    {
      if(!rhs_type.isa<RankedTensorType>()  && lhs_tensor_type.getElementType() == rhs_type)
      {
        auto scalarExpanded = rewriter.create<triton::SplatOp>(op->getLoc(), lhs.getType(), rhs);
        expandedOp = rewriter.create<T>(op->getLoc(), op->template getAttrOfType<TAttr>("predicate"), lhs, scalarExpanded);
      }
    }
    else if(auto rhs_tensor_type = rhs_type.dyn_cast_or_null<RankedTensorType>())
    {
      if(rhs_tensor_type.getElementType() == lhs_type)
      {
        auto scalarExpanded = rewriter.create<triton::SplatOp>(op->getLoc(), rhs.getType(), lhs);
        expandedOp = rewriter.create<T>(op->getLoc(), op->template getAttrOfType<TAttr>("predicate"), scalarExpanded, rhs);
      }          
    }
  }
  else if (lhs_type != res_type)
  {
    expandedOp = rewriter.create<T>(op->getLoc(), op->template getAttrOfType<TAttr>("predicate"), op->getOperand(0), op->getOperand(1));
  }

  if(expandedOp)
  {
    // rewriter.replaceAllUsesWith(op, expandedOp);
    op->replaceAllUsesWith(expandedOp);

    for(auto& u: expandedOp->getUses())
    {
      if(dyn_cast<triton::SplatOp>(u.getOwner()))
      {
        rewriter.replaceAllUsesWith(u.getOwner()->getResult(0), expandedOp->getResult(0));
        rewriter.eraseOp(u.getOwner());
      }
    }
    rewriter.eraseOp(op);

    return success();
  }

  return failure();
}

void iterateOperations(Operation *op, PatternRewriter& rewriter) {
  // Handle each operation
  // op->print(llvm::outs()); // Example: Print the operation

  // Recursively iterate over nested operations
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for(auto arg: block.getArguments())
      { 
        if(arg.getType().isa<IndexType>())
        {
          arg.setType(IntegerType::get(arg.getContext(), 32));
        }
      }
      
      for(auto constOp: llvm::make_early_inc_range(block.getOps<arith::ConstantIndexOp>()))
      {
        rewriter.setInsertionPoint(constOp);
        auto i32Op = rewriter.create<arith::ConstantIntOp>(constOp->getLoc(), constOp.value(), 32);
        rewriter.replaceAllUsesWith(constOp, i32Op);
        rewriter.eraseOp(constOp);
      }

      for (Operation &nestedOp : block.getOperations()) 
      {
        if (auto castOp = llvm::dyn_cast<arith::IndexCastOp>(nestedOp))
        {
          if(castOp.getIn().getType() == castOp.getOut().getType())
          {
            rewriter.replaceAllUsesWith(castOp, castOp.getIn());
            rewriter.eraseOp(castOp);
          }
        }
        if(!nestedOp.getRegions().empty())
        {
          iterateOperations(&nestedOp, rewriter);
        }
      }
    }
  }
}

void convertBlockArgTypes(mlir::triton::FuncOp func, PatternRewriter& rewriter) {
  // Iterate over top-level blocks in the function
  for (Block &block : func) {
    for(auto arg: block.getArguments())
      { 
        if(arg.getType().isa<IndexType>())
        {
          arg.setType(IntegerType::get(arg.getContext(), 32));
        }
      }
      
      for(auto constOp: llvm::make_early_inc_range(block.getOps<arith::ConstantIndexOp>()))
      {
        rewriter.setInsertionPoint(constOp);
        auto i32Op = rewriter.create<arith::ConstantIntOp>(constOp->getLoc(), constOp.value(), 32);
        rewriter.replaceAllUsesWith(constOp, i32Op);
        rewriter.eraseOp(constOp);
      }

      for (Operation &nestedOp : block.getOperations()) 
      {
        if (auto castOp = llvm::dyn_cast<arith::IndexCastOp>(nestedOp))
        {
          if(castOp.getIn().getType() == castOp.getOut().getType() || (castOp.getIn().getType().isInteger(32) && castOp.getOut().getType().isIndex()))
          {
            rewriter.replaceAllUsesWith(castOp, castOp.getIn());
            rewriter.eraseOp(castOp);
          }
        }
        if(!nestedOp.getRegions().empty())
        {
          iterateOperations(&nestedOp, rewriter);
        }
      }
  }
}

void isAncestor(mlir::Operation* pop, mlir::Value val, std::vector<std::vector<mlir::Operation*>>& chain, std::vector<mlir::Operation*> op_path)
{
  // bool res = false;
  op_path.push_back(pop);
  // std::vector<mlir::Operation*> path;
  for(auto op: pop->getOperands())
  {
    if (op == val)
    {
      chain.push_back(op_path);
    }
    else if (op.getDefiningOp())
    {
      isAncestor(op.getDefiningOp(), val, chain, op_path);
    }
  }
}

class RewriteReduction : public OpConversionPattern<mlir::scf::ForOp> {

public:
  using OpConversionPattern<mlir::scf::ForOp>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::scf::ForOp forOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
      
      if (forOp.getNumRegionIterArgs() != 0 || forOp->hasAttr("programs_loop_x") || forOp->hasAttr("programs_loop_y") || (forOp.getOps<mlir::triton::DotOp>().empty() && forOp.getOps<mlir::triton::ReduceOp>().empty())) 
      {
        return mlir::failure();
      }

    //   COMET_ERRS << forOp;

      std::vector<mlir::Value> operands;
      std::list<std::vector<std::vector<mlir::Operation*>>> ops_chain;
      std::vector<mlir::Value> yieldPtrs;
      std::set<void*> offsetedPtrs;
      std::vector<mlir::Value> basePtrs;
      int numLoadOps = -1;
    //   COMET_ERRS << forOp.getLowerBound();
    //   COMET_ERRS << forOp.getStep();
        std::vector<mlir::OpOperand*> maintain;
      if(mlir::isa<mlir::arith::ConstantIntOp>(forOp.getLowerBound().getDefiningOp()) && mlir::isa<mlir::arith::ConstantIntOp>(forOp.getStep().getDefiningOp()) && mlir::cast<mlir::arith::ConstantIntOp>(forOp.getStep().getDefiningOp()).value() == 1)
      {
        auto ops = forOp->getRegion(0).getOps<mlir::triton::LoadOp>();
        numLoadOps = 0;
        for(auto op: ops) 
        {
          std::vector<std::vector<mlir::Operation*>> local_ops_chain;
          std::vector<mlir::Operation*> ops_chain;
          mlir::Value offset =  op.getPtr().getDefiningOp()->getOperand(1);
          // mlir::Value blockPtr =  op.getPtr().getDefiningOp()->getOperand(0);
          if(offset == forOp.getLoopRegions()[0]->getArgument(0))
          {
            // COMET_ERRS << " SAME:\n";
            // COMET_ERRS << offset;
            // COMET_ERRS << forOp.getLoopRegions()[0]->getArgument(0);
            continue;
          }
          isAncestor(offset.getDefiningOp()->getOperand(0).getDefiningOp(), forOp.getLoopRegions()[0]->getArgument(0), local_ops_chain, ops_chain);
          mlir::Value base, step;
          mlir::Value res;
          std::vector<mlir::Value> stepsMul, stepsAdd;
          if(!local_ops_chain.empty())
          {
            base = offset.getDefiningOp()->getOperand(1);
          }
          else 
          {
            base = offset.getDefiningOp()->getOperand(0);
            local_ops_chain.clear();
            ops_chain.clear();
            isAncestor(offset.getDefiningOp()->getOperand(1).getDefiningOp(), forOp.getLoopRegions()[0]->getArgument(0), local_ops_chain, ops_chain);
          }

          if(local_ops_chain.empty())
          {
            continue;
          }
          numLoadOps++;

        //   COMET_ERRS <<"Chains\n";
          for(auto chain: local_ops_chain)
          {
            for(auto op: chain)
            {
            //   COMET_ERRS << op;
              if(mlir::isa<mlir::arith::MulIOp>(op))
              {
                stepsMul.push_back(op->getOperand(1));
              }

            }
          }
          for(auto s: stepsMul)
          {
            if(s.getType() != offset.getType())
            {
              if(auto sTensor = s.getType().dyn_cast<mlir::RankedTensorType>())
              {
                if(sTensor.getRank() != offset.getType().cast<mlir::RankedTensorType>().getRank())
                {
                  if(sTensor.getDimSize(0) == offset.getType().cast<mlir::RankedTensorType>().getDimSize(0))
                  {
                    s = rewriter.create<mlir::triton::ExpandDimsOp>(op->getLoc(), s, 1);
                  }
                  else 
                  {
                    s = rewriter.create<mlir::triton::ExpandDimsOp>(op->getLoc(), s, 0);
                  }
                }
                auto temp = rewriter.create<mlir::triton::BroadcastOp>(op->getLoc(), offset.getType(), s);
                
                if(!res)
                {
                  res = temp;
                }
                else
                {
                  res = rewriter.create<mlir::arith::MulIOp>(op->getLoc(), res, temp);
                }
              }
              else if (s.getType().isa<mlir::IntegerType>() )
              {
                auto temp = rewriter.create<mlir::triton::SplatOp>(op->getLoc(), offset.getType(), s);
                
                if(!res)
                {
                  res = temp;
                }
                else
                {
                  res = rewriter.create<mlir::arith::MulIOp>(op->getLoc(), res, temp);
                }
              }
            }
          }
          rewriter.setInsertionPoint(forOp);
        //   COMET_ERRS <<"Start From\n";
        //   COMET_ERRS << blockPtr;
        //   COMET_ERRS <<"+\n";
        //   COMET_ERRS << base;


          auto basePtr = op.getPtr(); //rewriter.create<mlir::triton::AddPtrOp>(op->getLoc(), op.getPtr().getType(), blockPtr, base)->getResult(0);
          op.getPtrMutable().assign(basePtr);

          basePtrs.push_back(basePtr);
          rewriter.setInsertionPointAfter(op);
          auto newPtr = rewriter.create<mlir::triton::AddPtrOp>(op->getLoc(), op.getPtr().getType(), basePtr, res)->getResult(0);
          yieldPtrs.push_back(newPtr);
        //   COMET_ERRS << newPtr;
        //   COMET_ERRS <<"Step is\n";
        //   COMET_ERRS << res;
        //   COMET_ERRS <<"+\n";
          // base.dump();

          for(auto chain: local_ops_chain)
          {
            for(auto op = chain.rbegin(); op != chain.rend(); op++)
            {
              if((*op)->getBlock() == forOp.getBody())
              {
                (*op)->moveBefore(forOp);
              }
            }
          }
          op.getPtr().getDefiningOp()->getOperand(1).getDefiningOp()->moveBefore(forOp);
          op.getPtr().getDefiningOp()->moveBefore(forOp);

          local_ops_chain.clear();
          ops_chain.clear();
          isAncestor(op.getMask().getDefiningOp()->getOperand(0).getDefiningOp(), forOp.getLoopRegions()[0]->getArgument(0), local_ops_chain, ops_chain);
          mlir::OpOperand* maskBase;
          if(!local_ops_chain.empty())
          {
            maskBase = &op.getMask().getDefiningOp()->getOpOperand(0);
          }
          else 
          {
            maskBase = &op.getMask().getDefiningOp()->getOpOperand(1);
            local_ops_chain.clear();
            ops_chain.clear();
            isAncestor(op.getMask().getDefiningOp()->getOperand(1).getDefiningOp(), forOp.getLoopRegions()[0]->getArgument(0), local_ops_chain, ops_chain);
          }

          for(auto chain: local_ops_chain)
          {
            rewriter.setInsertionPoint(maskBase->getOwner());

            mlir::IRMapping map;
            for(auto local_op = chain.rbegin(); local_op != chain.rend(); local_op++)
            {
              auto newvalue = rewriter.clone(**local_op, map);
              map.map(*local_op, newvalue);
              if(local_op == chain.rbegin())
              {
                for(auto& operand: newvalue->getOpOperands())
                {
                  if(operand.get() == forOp.getLoopRegions()[0]->getArgument(0))
                  {
                    maintain.push_back(&operand);
                  }
                }
              }
              // llvm::errs() << "Insreted\n";
              // newvalue->dump();
              maskBase->set(newvalue->getOpResult(0));
            }
          }
        }
      }
      else {
        // COMET_ERRS << "NOT RUNNING\n";
      }
      rewriter.setInsertionPoint(forOp);
      if(basePtrs.size() > 0 && numLoadOps > 0 &&  basePtrs.size() == (size_t)numLoadOps)
      {
        rewriter.replaceAllUsesWith(forOp.getLoopRegions()[0]->getArgument(0), rewriter.create<mlir::arith::ConstantIntOp>(forOp->getLoc(), 0, 32));
      }

      rewriter.setInsertionPoint(forOp);

      triton::DotOp dotOp;
      triton::ReduceOp reduceOp;
      if (!forOp.getOps<mlir::triton::DotOp>().empty())
      {
        dotOp = *forOp.getOps<mlir::triton::DotOp>().begin();
        auto elementType = mlir::isa<mlir::RankedTensorType>(dotOp->getResultTypes()[0]) ? dotOp->getResultTypes()[0].cast<mlir::RankedTensorType>().getElementType() : dotOp->getResultTypes()[0];
        auto init = rewriter.create<mlir::arith::ConstantOp>(forOp->getLoc(), elementType, rewriter.getZeroAttr(elementType));
        auto initSplat = rewriter.create<mlir::triton::SplatOp>(forOp->getLoc(), dotOp->getResultTypes()[0], init);
        dotOp.getCMutable().assign(initSplat);
        basePtrs.push_back(initSplat);
      }
      if (!forOp.getOps<mlir::triton::ReduceOp>().empty())
      {
        reduceOp = *forOp.getOps<mlir::triton::ReduceOp>().begin();
        auto elementType = mlir::isa<mlir::RankedTensorType>(reduceOp->getUsers().begin()->getResultTypes()[0]) ? reduceOp->getUsers().begin()->getResultTypes()[0].cast<mlir::RankedTensorType>().getElementType() : reduceOp->getUsers().begin()->getResultTypes()[0];
        auto init = rewriter.create<mlir::arith::ConstantOp>(forOp->getLoc(), elementType, rewriter.getZeroAttr(elementType));
        auto initSplat = rewriter.create<mlir::triton::SplatOp>(forOp->getLoc(), reduceOp->getUsers().begin()->getResultTypes()[0], init);
        basePtrs.push_back(initSplat);
      }
    //   COMET_ERRS << forOp;
      

      // mlir::Triton::DotOp

      auto ret = forOp.replaceWithAdditionalIterOperands(rewriter, {basePtrs}, true); 
      ret->getOperation()->setAttr("loop_block_size", forOp->getAttr("loop_block_size"));
      if(dotOp)
      {
        yieldPtrs.push_back(dotOp->getResult(0));
      }
      if(reduceOp)
      {
        yieldPtrs.push_back(reduceOp->getUsers().begin()->getResult(0));
      }

      auto yieldOp = *mlir::cast<mlir::scf::ForOp>(ret->getOperation()).getOps<mlir::scf::YieldOp>().begin();
      for(mlir::OpOperand* operand: maintain)
      {
        operand->set(mlir::cast<mlir::scf::ForOp>(ret->getOperation()).getLoopRegions()[0]->getArgument(0));
      }

      yieldOp->setOperands(yieldPtrs);
    //   COMET_ERRS << ret;

      if(dotOp)
      {
        auto dotOps = mlir::cast<mlir::scf::ForOp>(ret->getOperation()).getOps<mlir::triton::DotOp>();
        for(auto dotOp: dotOps)
        {
          for(auto u: dotOp->getUsers())
          {
            if (mlir::isa<mlir::triton::StoreOp>(u)) 
            {
              u->eraseOperand(1);
              u->insertOperands(1, ret->getLoopResults()->back());
              u->moveAfter(ret->getOperation());
            }
          }
        }
      }
      if(reduceOp)
      {
        auto reduceOps = mlir::cast<mlir::scf::ForOp>(ret->getOperation()).getOps<mlir::triton::ReduceOp>();
        assert(!reduceOps.empty());
        for(auto reduceOp: reduceOps)
        {
          assert(!reduceOp->getUsers().begin()->getUsers().empty());
          for(auto u: reduceOp->getUsers().begin()->getUsers())
          {
            // u->dump();
            if (mlir::isa<mlir::triton::StoreOp>(u)) 
            {
              u->eraseOperand(1);
              u->insertOperands(1, ret->getLoopResults()->back());
              u->moveAfter(ret->getOperation());
            }
          }
        }
      }
      // COMET_ERRS << ret->getOperation()->getParentOfType<mlir::ModuleOp>();

      return mlir::success();
    }

};


class FinalizeTritonFuncOp : public OpConversionPattern<mlir::triton::FuncOp> {

public:
  using OpConversionPattern<mlir::triton::FuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    bool changed = false;
    std::vector<Type> newArgTypes;
    for(auto arg: op.getArguments())
    {
      auto argT = arg.getType();

      if(argT.isa<IndexType>())
      {
        changed = true;
        arg.setType(rewriter.getI32Type());
      }
      else if(RankedTensorType t= argT.dyn_cast<RankedTensorType>())
      {
        if(t.getElementType().isa<IndexType>())
        {
          changed = true;
          arg.setType(RankedTensorType::get(t.getShape(), rewriter.getI32Type()));
        }
      }
    }

    if(!changed)
    {
      return failure();
    }

    
    for(auto arg: op.getArguments())
    {
      newArgTypes.push_back(arg.getType());
    }
    rewriter.startRootUpdate(op);
    auto functype= mlir::FunctionType::get(getContext(), newArgTypes, {});
    op.setType(functype);
    convertBlockArgTypes(op, rewriter);
    rewriter.finalizeRootUpdate(op);
    
    return success();
  }
};


template <typename T, typename TAttr>
class ArithCmpOpPattern : public OpConversionPattern<T> {
public:
  using OpConversionPattern<T>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return ExpandScalarTensorArithCmpOp<T, TAttr>(op, rewriter);
  }
};
class ArithSelectPattrn : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::SelectOp op, typename arith::SelectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    return ExpandScalarTensorSelectOp(op, rewriter);
  }
};

template <typename T>
class ArithOpPattern : public OpConversionPattern<T> {
public:
  using OpConversionPattern<T>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return ExpandScalarTensorArithOp(op, rewriter);
  }
};


template <>
class ArithOpPattern<arith::MulFOp> : public OpConversionPattern<arith::MulFOp> {
public:
  using OpConversionPattern<arith::MulFOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::MulFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if(op->hasAttr("dot") && op->getAttrOfType<BoolAttr>("dot"))
    {
      auto addOp = cast<arith::AddFOp>(*op->getUsers().begin());
      auto loadRes = addOp.getLhs() == op ? addOp.getRhs() : addOp.getLhs();
      rewriter.setInsertionPointAfter(addOp);
      auto lhsTensor = op->getOperand(0).getType().cast<RankedTensorType>();
      auto rhsTensor = op->getOperand(1).getType().cast<RankedTensorType>();
      mlir::Operation* dotOp;
      if (lhsTensor.getRank() == 2 && (rhsTensor.getRank() == 1 || (rhsTensor.getRank() == 2 && (rhsTensor.getDimSize(0) == 1 || rhsTensor.getDimSize(1) == 1)) ))
      {
        if (rhsTensor.getRank() == 1)
        {
          // auto expandOp = rewriter.create<triton::ExpandDimsOp>(op.getLoc(), op->getOperand(1), 0);
          // auto bcastOp = rewriter.create<triton::BroadcastOp>(op.getLoc(), RankedTensorType::get({lhsTensor.getDimSize(0),rhsTensor.getDimSize(1)}  , lhsTensor.getElementType()), expandOp);
          auto lhs = op->getOperand(0);
          auto rhs = op->getOperand(1);
          makeShapesEqual(op, lhs, rhs, rewriter);
          auto mulOp = rewriter.create<arith::MulFOp>(op.getLoc(), op->getOperand(0), rhs);
          auto sumOp = rewriter.create<triton::ReduceOp>(op->getLoc(), ValueRange(mulOp), 1);
          std::vector<Type> arg_types = {lhsTensor.getElementType(), lhsTensor.getElementType()};
          std::vector locs = {op->getOperand(0).getLoc(), op->getOperand(1).getLoc()};
          auto type_range = ArrayRef(arg_types);
          sumOp.getBodyRegion().emplaceBlock().addArguments(TypeRange(type_range), locs);
          rewriter.setInsertionPointToStart(sumOp.getBody(0));
          auto res = rewriter.create<arith::AddFOp>(sumOp->getLoc(), sumOp.getBody(0)->getArgument(0), sumOp.getBody(0)->getArgument(1));
          rewriter.create<triton::ReduceReturnOp>(res->getLoc(), ValueRange(res));
          rewriter.setInsertionPointAfter(sumOp);
          auto expandOp = rewriter.create<triton::ExpandDimsOp>(op.getLoc(), sumOp->getResult(0), 1);

          dotOp = expandOp;
          
        }
        else {
          return failure();
        }
      }
      else {
        dotOp = rewriter.create<triton::DotOp>(op->getLoc(), op->getOperand(0), op->getOperand(1), loadRes, rewriter.getBoolAttr(true), rewriter.getI32IntegerAttr(1));
      }
      // if(op->getOperand(1).getType().cast<RankedTensorType>().)

      // addOp->replaceAllUsesWith(dotOp);
      rewriter.replaceAllUsesWith(addOp, dotOp->getResult(0));

      for(auto& u: dotOp->getUses())
      {
        if(dyn_cast<triton::SplatOp>(u.getOwner()))
        {
          rewriter.replaceAllUsesWith(u.getOwner()->getResult(0), dotOp->getResult(0));
          rewriter.eraseOp(u.getOwner());
        }
      }
      
      // op->getParentOfType<ModuleOp>().dump();
      rewriter.eraseOp(addOp);
      rewriter.eraseOp(op);

      return success();
    }

    return ExpandScalarTensorArithOp(op, rewriter);
  }
};

class SimplifyAffineApply : public OpConversionPattern<affine::AffineApplyOp> {
public:
  using OpConversionPattern<affine::AffineApplyOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(affine::AffineApplyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    auto affineMap = op.getAffineMap();
    auto newAffine = create_simplified_affine_apply_op(op, rewriter);
    auto newMap = newAffine.getAffineMap();

    return success(newMap != affineMap);
  }
};


class MemrefLoadOpToTriton : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    return convertMemoryOp(op, rewriter);
  }
};

class MemrefStoreOpToTriton : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    return convertMemoryOp(op, rewriter);
  }
};

class GpuFuncOpToTritonFuncOp : public OpConversionPattern<mlir::gpu::GPUFuncOp> {
private: 
  std::map<std::string, int>& kernel_names;
public:

  GpuFuncOpToTritonFuncOp(const TypeConverter &typeConverter, MLIRContext *context, std::map<std::string, int>& kernel_names) : OpConversionPattern(typeConverter, context), kernel_names(kernel_names) {} 
  using OpConversionPattern<mlir::gpu::GPUFuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::gpu::GPUFuncOp gpuFuncOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // COMET_ERRS << "GpuFuncOpToTritonFuncOp" << "\n"; 
    if (gpuFuncOp->getAttr("copied"))
    {
    //   COMET_ERRS << "Found copied in GpuFuncOpToTritonFuncOp" << "\n"; 
      return mlir::failure();
    }

    std::string orig_name = gpuFuncOp.getName().str();

    std::vector<NamedAttribute> named_attrs;
    named_attrs.push_back(rewriter.getNamedAttr("copied", mlir::UnitAttr::get(gpuFuncOp->getContext())));
    named_attrs.push_back(rewriter.getNamedAttr("block_size_x", gpuFuncOp->getAttr("block_size_x")));
    named_attrs.push_back(rewriter.getNamedAttr("block_size_y", gpuFuncOp->getAttr("block_size_y")));
    named_attrs.push_back(rewriter.getNamedAttr("gpu.kernel", rewriter.getUnitAttr()));

    // We need to change the name of the original function to avoid mlir throwing an error for two functions with the same name
    gpuFuncOp.setName(orig_name + "_deleted");

    auto newGpuFunc = rewriter.create<mlir::gpu::GPUFuncOp>(rewriter.getUnknownLoc(), orig_name, gpuFuncOp.getFunctionType(), TypeRange(), TypeRange(), named_attrs);
    rewriter.setInsertionPointToEnd(&newGpuFunc.getFunctionBody().front());
    rewriter.create<mlir::gpu::ReturnOp>(newGpuFunc->getLoc());
    rewriter.setInsertionPoint(gpuFuncOp);

    std::vector<Type> newArgTypes;
    std::vector<std::pair<size_t,Type>> ttypes;
    
    // Convert input arguments to Triton types (mainly memref-> ptr)
    size_t offset = 0;
    auto typeConverter = getTypeConverter<mlir::comet::GpuTypeConverter>();
    for(size_t ii = 0; ii < gpuFuncOp.getNumArguments(); ii++ )
    {
      Type type = gpuFuncOp.getArgument(ii).getType();
      SmallVector<Type,3> types;
      auto is_memref = typeConverter->convertType(type, types);
      if (is_memref.succeeded())
      {
        newArgTypes.push_back(types[0]);
        for(size_t i = 1; i < types.size(); i++)
        {
          ttypes.push_back(std::make_pair(offset + ii + i,types[i]));
        }
        offset += types.size() -1;
      }
      else 
      {
        auto newType = typeConverter->convertType(type);
        newArgTypes.push_back(newType);
      }
    }
    
    auto functype= mlir::FunctionType::get(getContext(), newArgTypes, {});
    rewriter.setInsertionPointToEnd(gpuFuncOp->getParentOfType<ModuleOp>().getBody());

    if (kernel_names.find(orig_name) != kernel_names.end())
    {
      kernel_names[orig_name] ++;
    } 
    else {
      kernel_names[orig_name] = 0;
    }

    // Replace the gpu.func with a tt.func that has the new input args
    auto TTFunc = rewriter.create<mlir::triton::FuncOp>(gpuFuncOp->getLoc(), orig_name+std::to_string(kernel_names[orig_name]), functype);
    TTFunc->setAttr("origin", rewriter.getStringAttr(gpuFuncOp->getParentOfType<mlir::gpu::GPUModuleOp>().getName() + "::" +orig_name) );
    TTFunc->setAttr("block_size_x", gpuFuncOp->getAttr("block_size_x") );
    TTFunc->setAttr("block_size_y", gpuFuncOp->getAttr("block_size_y") );
    auto newFuncBody = TTFunc.addEntryBlock();
    rewriter.setInsertionPointToEnd(newFuncBody);

    auto newFuncArgs = TTFunc.getBody().getArguments();
    llvm::SmallVector<Location, 4> locs;
    for(auto arg: newFuncArgs)
    {
      locs.push_back(arg.getLoc());
    }

    Block* new_block = rewriter.createBlock(&gpuFuncOp.getFunctionBody().front(), gpuFuncOp.getFunctionBody().front().getArgumentTypes(), locs);
    rewriter.setInsertionPointToEnd(new_block);
    rewriter.create<mlir::gpu::ReturnOp>(rewriter.getUnknownLoc());


    rewriter.startRootUpdate(TTFunc);
    for(auto arg : llvm::zip(gpuFuncOp.getFunctionBody().back().getArguments(), TTFunc.getFunctionBody().back().getArguments()))
    {
      std::get<0>(arg).replaceAllUsesWith(std::get<1>(arg));
    }

    rewriter.finalizeRootUpdate(TTFunc);
    rewriter.mergeBlocks(&gpuFuncOp.getFunctionBody().back(), &TTFunc.getFunctionBody().front(), TTFunc.getFunctionBody().front().getArguments());
    rewriter.eraseOp(gpuFuncOp);
    
    // Reenable for bound guards
    // for(auto it =  ttypes.begin(); it != ttypes.end(); ++it)
    // {
    //   newFunc.insertArgument(it->first, it->second, {}, op->getLoc());
    // }
    
    //Insert program count for each dimension (Y,X)
    TTFunc.insertArgument(0, rewriter.getIndexType(), {}, gpuFuncOp->getLoc());
    TTFunc.insertArgument(0, rewriter.getIndexType(), {}, gpuFuncOp->getLoc());


    // Remove the func.ret Op
    for(auto &o: TTFunc.getOps())
    {
      if(isa<mlir::gpu::ReturnOp>(o))
      {
        rewriter.eraseOp(&o);
      }
    }

    rewriter.setInsertionPointToStart(newFuncBody);


    auto funcOps = TTFunc.getOps();
    std::vector<Operation*> ops;
    for(auto& op: funcOps)
    {
      ops.push_back(&op);
    }
    rewriter.setInsertionPointToStart(&TTFunc.getFunctionBody().front());

    // To handle cases where the number of thread-blocks is not big enough to cover the data space, we introduce an extra set of loops
    // that will make each block run again after a num-blocks stride
    // e.g. if we have an array with size N > 65k  and map a single block per column then we are limited by CUDA architecture's 65k blocks
    // Thus, we can pass the real number of blocks that we need and the first N%65k thread-blocks will run for the elements beyond the column 65k 
    mlir::Value numProgramsY = rewriter.create<triton::GetNumProgramsOp>(TTFunc->getLoc(), 1);
    mlir::Value numProgramsX = rewriter.create<triton::GetNumProgramsOp>(TTFunc->getLoc(), 0);
    mlir::Value myIdY = rewriter.create<triton::GetProgramIdOp>(TTFunc->getLoc(),  mlir::triton::ProgramIDDimAttr::get(gpuFuncOp->getContext(), mlir::triton::ProgramIDDim::Y));
    mlir::Value myIdX = rewriter.create<triton::GetProgramIdOp>(TTFunc->getLoc(),  mlir::triton::ProgramIDDimAttr::get(gpuFuncOp->getContext(), mlir::triton::ProgramIDDim::X));

    mlir::Value indexNumProgramsY = rewriter.create<arith::IndexCastOp>(numProgramsY.getLoc(), rewriter.getIndexType(), numProgramsY);
    mlir::Value indexNumProgramsX = rewriter.create<arith::IndexCastOp>(numProgramsX.getLoc(), rewriter.getIndexType(), numProgramsX);
    
    mlir::Value indexMyIdY = rewriter.create<arith::IndexCastOp>(myIdY.getLoc(), rewriter.getIndexType(), myIdY);
    mlir::Value indexMyIdX = rewriter.create<arith::IndexCastOp>(myIdX.getLoc(), rewriter.getIndexType(), myIdX);

    mlir::Value boundY = rewriter.create<arith::SubIOp>(myIdY.getLoc(), TTFunc.getArgument(0), indexMyIdY);
    mlir::Value boundX = rewriter.create<arith::SubIOp>(myIdX.getLoc(), TTFunc.getArgument(1), indexMyIdX);

    mlir::Value zero = rewriter.create<arith::ConstantIndexOp>(myIdY.getLoc(), 0);


    mlir::scf::ForOp yLoop = rewriter.create<mlir::scf::ForOp>(TTFunc->getLoc(), zero, boundY, indexNumProgramsY);
    yLoop->setAttr("programs_loop_y", rewriter.getUnitAttr());
    rewriter.setInsertionPointToStart(yLoop.getBody());
    mlir::scf::ForOp xLoop = rewriter.create<mlir::scf::ForOp>(TTFunc->getLoc(), zero, boundX, indexNumProgramsX);
    xLoop->setAttr("programs_loop_x", rewriter.getUnitAttr());
    // bool foundYieldOp = false;

    auto offsetY = yLoop.getBody()->getArgument(0);
    auto offsetX = xLoop.getBody()->getArgument(0);
    

    for(auto op : ops)
    {
      op->moveBefore(xLoop.getBody()->getTerminator());
    }

    auto block_ids = xLoop.getOps<mlir::gpu::BlockIdOp>();

    auto expr = mlir::getAffineDimExpr(0, rewriter.getContext()) + mlir::getAffineDimExpr(1, rewriter.getContext());
    auto affineIndex = mlir::AffineMap::get(2, 0, {expr}, rewriter.getContext());
    rewriter.setInsertionPointToStart(xLoop.getBody());
    
    for(auto bId: block_ids)
    {
        rewriter.setInsertionPointAfter(bId);
        if(bId.getDimension() == mlir::gpu::Dimension::x)
        {
          std::vector<mlir::Value> values;
          values.push_back(bId);
          values.push_back(offsetX);
          auto newBlockIdX = rewriter.create<mlir::affine::AffineApplyOp>(gpuFuncOp->getLoc(), affineIndex, values);
          rewriter.replaceAllUsesExcept(bId, newBlockIdX, newBlockIdX);
        }
        else if(bId.getDimension() == mlir::gpu::Dimension::y)
        {
          std::vector<mlir::Value> values;
          values.push_back(bId);
          values.push_back(offsetY);
          auto newBlockIdY = rewriter.create<mlir::affine::AffineApplyOp>(gpuFuncOp->getLoc(), affineIndex, values);
          rewriter.replaceAllUsesExcept(bId, newBlockIdY, newBlockIdY);
        }
    }

    rewriter.setInsertionPointToEnd(&TTFunc.getFunctionBody().front());
    rewriter.create<triton::ReturnOp>(TTFunc.getLoc());

    // COMET_ERRS << "Done with funcOp convertion\n";
    // COMET_ERRS << gpuFuncOp->getParentOfType<ModuleOp>();
    // TTFunc->getParentOfType<ModuleOp>()->dump();
    return success();
  }
};

class BlockifyReduction : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto block_ids = op.getOps<mlir::gpu::BlockIdOp>();
    auto parent = op->getParentOfType<scf::ForOp>();
    while(parent != NULL && !parent->hasAttr("programs_loop_x"))
    {
      parent = parent->getParentOfType<scf::ForOp>();
    }
    if (parent == NULL)
    {
        return mlir::failure();
    }

    auto offsetX = parent.getBody()->getArgument(0);
    auto offsetY = parent->getParentOfType<scf::ForOp>().getBody()->getArgument(0);
    
    for(auto bID: block_ids)
    {
      if(bID.getDimension() == mlir::gpu::Dimension::x)
      {
        auto newBlockIdX = rewriter.create<arith::AddIOp>(op->getLoc(), bID, offsetX);
        rewriter.replaceAllUsesExcept(bID, newBlockIdX, newBlockIdX);
      }
      else if(bID.getDimension() == mlir::gpu::Dimension::y)
      {
        auto newBlockIdY = rewriter.create<arith::AddIOp>(op->getLoc(), bID, offsetY);
        rewriter.replaceAllUsesExcept(bID, newBlockIdY, newBlockIdY);
      }
    }


    if (!op->hasAttr("loop_block_size") && op->hasAttr("reduceDim") && op->getAttrOfType<StringAttr>("reduceDim").str().compare("dimR_grid") == 0)
    {
      auto forOpBlock = *op.getOps<scf::ForOp>().begin();
      
      auto newForOp = rewriter.create<scf::ForOp>(op->getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep());
      if(auto block_size_r = llvm::cast_if_present<arith::ConstantIndexOp>(forOpBlock.getUpperBound().getDefiningOp()))
      {
        newForOp->setAttr("loop_block_size", rewriter.getIndexAttr(block_size_r.value()));
      }
      else if(auto input_block_size = llvm::cast_if_present<BlockArgument>(forOpBlock.getUpperBound())){
        auto origin_kernel_fullname = cast<triton::FuncOp>(input_block_size.getOwner()->getParentOp())->getAttrOfType<StringAttr>("origin").str();
        auto origin_kernel_module_name = origin_kernel_fullname.substr(0, origin_kernel_fullname.find(":"));
        // COMET_ERRS << "origin_kernel_module_name: " << origin_kernel_module_name <<"\n";
        auto origin_kernel_name = origin_kernel_fullname.substr(origin_kernel_fullname.find(":") + 2);
        // COMET_ERRS << "origin_kernel_name: " << origin_kernel_name <<"\n";
        auto funcOp = *op->getParentOfType<ModuleOp>().getOps<mlir::func::FuncOp>().begin();
        // COMET_ERRS << "Searching " << funcOp << " for launchFuncOps: \n";
        for(auto launchOp : funcOp.getOps<mlir::gpu::LaunchFuncOp>())
        {
          // COMET_ERRS << launchOp.getKernelModuleName() <<" vs " << origin_kernel_module_name << "\n";
          // COMET_ERRS << launchOp.getKernelName() <<" vs " << origin_kernel_name << "\n";
          if(launchOp.getKernelModuleName().strref().equals(origin_kernel_module_name) && launchOp.getKernelName().strref().equals(origin_kernel_name))
          {
            // COMET_ERRS << "FOUND LAUNCHOP" << origin_kernel_fullname << "\n";
            auto block_size_r = cast<arith::ConstantIndexOp>(launchOp.getKernelOperand(input_block_size.getArgNumber()-2).getDefiningOp()).value();
            newForOp->setAttr("loop_block_size", rewriter.getIndexAttr(block_size_r));
            // COMET_ERRS << "Chaning forOP" << newForOp << "\n";
          }
        }
      }

      rewriter.eraseOp(*newForOp.getBody()->getOps<mlir::scf::YieldOp>().begin());
      rewriter.setInsertionPointToStart(op.getBody());
      
      for(auto arg: llvm::zip(op.getBody()->getArguments(), newForOp.getBody()->getArguments()))
      {
        std::get<0>(arg).replaceAllUsesWith(std::get<1>(arg));
      }
      rewriter.mergeBlocks(op.getBody(), newForOp.getBody(), newForOp.getBody()->getArguments());
      rewriter.eraseOp(op);
      // COMET_ERRS << newForOp;
      // COMET_ERRS << op->getParentOfType<ModuleOp>();

      return success();
    }
    else if( op->hasAttr("reduceDim") && op->getAttrOfType<StringAttr>("reduceDim").str().compare("dimR_block") == 0) 
    {
      for(auto store: op.getBody()->getOps<memref::StoreOp>())
      {
        bool reduce = true;
        for(auto oper: store.getOperands())
        {
          if (isa<MemRefType>(oper.getType()))
          {
            continue;
          }
          else if (oper == op.getBody()->getArgument(0)) {
            reduce = false;
          }
          if (reduce)
          {
            auto toStore = store.getValueToStore();
            if(toStore.getDefiningOp() && isa<arith::AddFOp>(toStore.getDefiningOp()))
            {
              bool condition1 = isa<arith::MulFOp>(toStore.getDefiningOp()->getOperand(1).getDefiningOp()) && isa<memref::LoadOp>(toStore.getDefiningOp()->getOperand(0).getDefiningOp());
              bool condition2 = isa<arith::MulFOp>(toStore.getDefiningOp()->getOperand(0).getDefiningOp()) && isa<memref::LoadOp>(toStore.getDefiningOp()->getOperand(1).getDefiningOp());
              if (condition1 || condition2)
              {
                if (cast<memref::LoadOp>(toStore.getDefiningOp()->getOperand(condition1 ? 0 : 1).getDefiningOp()).getMemRef() == store.getMemRef())
                {
                  toStore.getDefiningOp()->getOperand(condition1 ? 1 : 0).getDefiningOp()->setAttr("dot", rewriter.getBoolAttr(true));
                }
              }
            }
          }
        }
      }

      rewriter.setInsertionPoint(op);
      auto icast = rewriter.create<arith::IndexCastOp>(op->getLoc(), rewriter.getI32Type(), op->getBlock()->getArguments()[0]);
      icast->setAttr("ReductionIndex", rewriter.getUnitAttr());
      //[TODO] auto redIdx = rewriter.create<IntToRedIndexOp>(op.getLoc(), icast);

      rewriter.eraseOp(*op.getBody()->getOps<mlir::scf::YieldOp>().begin());
      //[TODO] rewriter.replaceAllUsesWith(op.getBody()->getArgument(0), redIdx);
      //[TODO] rewriter.inlineBlockBefore(op.getBody(), op, redIdx.getResult());
      
      // [TODO] Remove the following
      rewriter.replaceAllUsesWith(op.getBody()->getArgument(0), icast);
      rewriter.inlineBlockBefore(op.getBody(), op, icast.getResult());
      // COMET_ERRS << op->getParentOfType<ModuleOp>();
      rewriter.eraseOp(op);

      return success();

    }
    std::cout << "Didn't match after\n";

    return failure();
  }
};


class ConvertGpuToTriton
    : public ConvertGpuKernelToTritonPassBase<ConvertGpuToTriton> {
public:
  ConvertGpuToTriton() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp op = getOperation();
    
    mlir::comet::GpuTypeConverter typeConverter(context);
    // typeConverter.blockX = blockX.getValue();
    // typeConverter.blockY = blockY.getValue();
    // typeConverter.blockR = blockR.getValue();
    mlir::comet::GpuConversionTarget target(getContext(), typeConverter);
    RewritePatternSet patterns(context);
    
    auto funcOp  = *op.getOps<mlir::func::FuncOp>().begin();
    auto launchOps = funcOp.getOps<mlir::gpu::LaunchFuncOp>();
    // bool changed = false;
    OpBuilder builder = OpBuilder(op);
    auto gpuModules = op.getOps<mlir::gpu::GPUModuleOp>();

    for(auto launchOp: launchOps)
    {
    //   COMET_ERRS << "RUNNING\n"; 
      if(launchOp->hasAttr("checked"))
      {
        continue;
      }
      // changed = true;
      auto kernel_name = launchOp.getKernelName();
      auto kernel_mod_name = launchOp.getKernelModuleName();
    //   COMET_ERRS <<kernel_mod_name <<"::"<< kernel_name << "\n";

      auto kernel_module = *std::find_if(gpuModules.begin(), gpuModules.end(), [&kernel_mod_name] (mlir::gpu::GPUModuleOp modOp) -> bool {return modOp.getName().equals(kernel_mod_name); });
      auto gpufuncOps = kernel_module.getOps<mlir::gpu::GPUFuncOp>(); 
      auto gpufuncOp = *std::find_if(gpufuncOps.begin(), gpufuncOps.end(), [&kernel_name](mlir::gpu::GPUFuncOp funcOp) -> bool{ return funcOp->getAttrOfType<mlir::StringAttr>("sym_name").strref().equals(kernel_name) ;});
      auto b_size_x = dyn_cast<arith::ConstantIndexOp>(launchOp.getBlockSizeX().getDefiningOp()).value();
      auto b_size_y = dyn_cast<arith::ConstantIndexOp>(launchOp.getBlockSizeY().getDefiningOp()).value();
      if (gpufuncOp->hasAttr("simplified"))
      {
        if((gpufuncOp->hasAttr("block_size_x") && gpufuncOp->getAttrOfType<IntegerAttr>("block_size_x").getValue() !=  b_size_x) || (gpufuncOp->hasAttr("block_size_y") && gpufuncOp->getAttrOfType<IntegerAttr>("block_size_y").getValue() !=  b_size_y))
        {
          auto new_func_op = gpufuncOp.clone();
          auto new_func_name = new_func_op.getName()+"_"+std::to_string(b_size_x)+"_"+std::to_string(b_size_y);
          new_func_op.setName(new_func_name.str());
          kernel_module.push_back(new_func_op);
        }
        continue;
      }
      gpufuncOp->setAttr("simplified", builder.getUnitAttr());
      gpufuncOp->setAttr("block_size_x", builder.getIndexAttr(b_size_x));
      gpufuncOp->setAttr("block_size_y", builder.getIndexAttr(b_size_y));

      auto bitvector = llvm::BitVector(gpufuncOp.getNumArguments(), false);

      for(auto k_op : llvm::enumerate(launchOp.getKernelOperands()))
      {
        if(arith::ConstantIndexOp k = llvm::dyn_cast_if_present<arith::ConstantIndexOp>(k_op.value().getDefiningOp()))
        {
          // if(k.value() == 0 || k.value() == 1)
          // {
          //   bitvector.set(k_op.index());
          //   builder.setInsertionPointToStart(&gpufuncOp.getFunctionBody().getBlocks().front());
          //   gpufuncOp.getFunctionBody().getArgument(k_op.index()).replaceAllUsesWith(builder.create<arith::ConstantIndexOp>(gpufuncOp->getLoc(), k.value()));
          // }
          // else if(k.value() == b_size_x || k.value() == b_size_y)
          // {
          //   bitvector.set(k_op.index());
          //   builder.setInsertionPointToStart(&funcOp.getFunctionBody().getBlocks().front());
          //   auto k_val = (k.value() == b_size_x) ? b_size_x : b_size_y;
          //   funcOp.getFunctionBody().getArgument(k_op.index()).replaceAllUsesWith(builder.create<arith::ConstantIndexOp>(funcOp->getLoc(), k_val));
          // }
    //     else
    //     {
    //       break;
        }
      }
    
      launchOp->setAttr("checked", builder.getUnitAttr());
      int offset = 0;
      for(auto it = bitvector.set_bits_begin(); it != bitvector.set_bits_end(); ++it)
      {
        launchOp.getKernelOperandsMutable().erase(*it -offset++);
      } 
      
      gpufuncOp.eraseArguments(bitvector);
    }
    std::map<std::string, int> kernel_names;
    patterns.insert<BlockifyReduction>(typeConverter, context);
    // patterns.insert<GScfIfPattern>(typeConverter, context);
    patterns.insert<GpuFuncOpToTritonFuncOp>(typeConverter, context, kernel_names);
    patterns.insert<MemrefLoadOpToTriton, MemrefStoreOpToTriton>(context);
    patterns.insert<SimplifyAffineApply>(context);
    patterns.insert<ArithOpPattern<arith::AddFOp>, ArithOpPattern<arith::MulFOp>, ArithOpPattern<arith::SubFOp>, ArithOpPattern<arith::SubIOp>, ArithCmpOpPattern<arith::CmpFOp, arith::CmpFPredicateAttr>, ArithSelectPattrn, ArithCmpOpPattern<arith::CmpIOp, arith::CmpIPredicateAttr>>(context);
    mlir::arith::populateArithExpandOpsPatterns(patterns);

    std::vector<Operation*>  gpuModuleOps;
    op->walk([&gpuModuleOps](mlir::gpu::GPUModuleOp gMod) {gpuModuleOps.push_back(gMod); });

    if (failed(applyPartialConversion(gpuModuleOps, target, std::move(patterns))))
    {
      return signalPassFailure();
    }


    auto ttfuncOps  = op.getOps<mlir::triton::FuncOp>();

    for(auto ttfuncOp: ttfuncOps)
    {
      ttfuncOp->walk([](mlir::arith::MinUIOp op){if(op->hasAttr("GuardX")) {op->erase();}});
      ttfuncOp->walk([](mlir::arith::MinUIOp op){if(op->hasAttr("GuardY")) {op->erase();}});
      ttfuncOp->walk([](mlir::arith::MinUIOp op){if(op->hasAttr("GuardR")) {op->erase();}});
      ttfuncOp->walk([](mlir::affine::AffineApplyOp op){op->erase();});
    }


    RewritePatternSet patterns2(context);

    mlir::comet::GpuTypeConverter2 typeConverter2(context);
    mlir::comet::GpuConversionTarget2 target2(getContext(), typeConverter2);

    patterns2.insert<FinalizeTritonFuncOp>(context);
    patterns2.insert<ArithOpPattern<arith::AddFOp>, ArithOpPattern<arith::MulFOp>, ArithOpPattern<arith::SubFOp>, ArithOpPattern<arith::SubIOp>, ArithCmpOpPattern<arith::CmpFOp, arith::CmpFPredicateAttr>, ArithSelectPattrn, ArithCmpOpPattern<arith::CmpIOp, arith::CmpIPredicateAttr>>(context);

    if (failed(applyPartialConversion(op, target2, std::move(patterns2))))
    {
      // COMET_ERRS << "Failed to Lower STCOutputLowering2\n";
      return signalPassFailure();
    }


    PassManager pm(op.getContext());


    // Add the LoopInvariantCodeMotion pass to the pass manager
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    // Run the pass manager
    if (failed(pm.run(op))) {
      signalPassFailure();
      return;

    }

    RewritePatternSet patterns3(context);
    ConversionTarget target3(*context);
    target3.addLegalDialect<arith::ArithDialect, triton::TritonDialect>();
    target3.addIllegalDialect<scf::SCFDialect>();
    target3.addLegalOp<scf::YieldOp>();
    target3.addDynamicallyLegalOp<scf::ForOp>([](scf::ForOp forOp) {

        if (forOp->hasAttr("loop_block_size") && forOp.getNumRegionIterArgs() != 0)
        {
          return true;
        }
        else if (forOp->hasAttr("programs_loop_x") || forOp->hasAttr("programs_loop_y"))
        {
          return true;
        }
        return false;
    });
    
    patterns3.insert<RewriteReduction>(context);
    for(auto ttfuncOp: op.getOps<triton::FuncOp>())
    {
      if (failed(applyPartialConversion(ttfuncOp, target3, std::move(patterns3))))
      {
        // COMET_ERRS << "Failed to Lower STCOutputLowering2\n";
        signalPassFailure();
        return ;
      }
    }

    // Add the LoopInvariantCodeMotion pass to the pass manager
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    // Run the pass manager
    if (failed(pm.run(op))) {
      signalPassFailure();
      return;

    }
  }
};

}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::comet::createConvertGpuKernelToTritonPass() {
  return std::make_unique<::ConvertGpuToTriton>();
}