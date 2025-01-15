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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/IndexedMap.h"
#include <cstdint>

#include "comet/Dialect/IndexTree/IR/IndexTreeDialect.h"
#include "comet/Dialect/TensorAlgebra/IR/TADialect.h"
#include "comet/Dialect/IndexTree/Passes.h"

using namespace mlir;
using namespace mlir::indexTree;
using namespace mlir::tensorAlgebra;

namespace mlir {
  namespace comet{
    #define GEN_PASS_DEF_INDEXTREESYMBOLICCOMPUTEPASS
    #include "comet/Dialect/IndexTree/Passes.h.inc"
  }
}

struct CreateSymbolicTree :  public OpRewritePattern<IndexTreeSparseTensorOp> {
  CreateSymbolicTree(MLIRContext *context)
      : OpRewritePattern<IndexTreeSparseTensorOp>(context, /*benefit=*/1) {}

  Value copyDomain(Value domain, mlir::PatternRewriter &rewriter, 
                   Location loc, IRMapping& map, 
                   llvm::SmallDenseMap<std::pair<Value, int32_t>, Value>& tensor_to_node,
                   Value* parent,
                   Value predecessor = nullptr) const
  {
    Value new_domain;
    Operation* domain_op = domain.getDefiningOp();
    if(llvm::isa<IndexTreeDomainIntersectionOp>(domain_op))
    {
      auto intersection_domain_op = llvm::cast<IndexTreeDomainIntersectionOp>(domain_op);
      for(Value subdomain : intersection_domain_op.getDomains()){
        copyDomain(subdomain, rewriter, loc, map, tensor_to_node, parent, predecessor);
      }
    }
    else if(llvm::isa<IndexTreeDomainUnionOp>(domain_op))
    {
      auto union_domain_op = llvm::cast<IndexTreeDomainUnionOp>(domain_op);
      for(Value subdomain : union_domain_op.getDomains()){
        copyDomain(subdomain, rewriter, loc, map, tensor_to_node, parent, predecessor);
      }
    }
    
    if(llvm::isa<IndexTreeNestedDomainOp>(domain_op)){
      auto nested_domain = llvm::cast<IndexTreeNestedDomainOp>(domain_op);
      for(auto subdomain_itr = nested_domain.getDomains().begin(); subdomain_itr != nested_domain.getDomains().end();)
      {
        Value subdomain = *subdomain_itr;
        new_domain = copyDomain(subdomain, rewriter, loc, map, tensor_to_node, parent, predecessor);
        subdomain_itr++;
        if(subdomain_itr != nested_domain.getDomains().end())
        {
          auto index_node_type = indexTree::IndexNodeType::get(rewriter.getContext());
          indexTree::IndexTreeIndicesOp index_node_op = rewriter.create<indexTree::IndexTreeIndicesOp>(loc, index_node_type, *parent, new_domain);
          createMapping(index_node_op, subdomain, tensor_to_node);
          *parent = index_node_op.getOutput();
          predecessor = *parent;
        } else {
          map.map(nested_domain, new_domain);
        }
      }
    } else if(llvm::isa<IndexTreeSparseDomainOp>(domain_op))
    {
      auto sparse_domain_op = llvm::cast<IndexTreeSparseDomainOp>(domain_op);
      auto tensor = sparse_domain_op.getTensor();
      int32_t dim = sparse_domain_op.getDim();
      Value parent = nullptr;
      if(dim > 0){
        // TODO: Determine parent of this op
        // Will be needed for 3 dimensional sparse tensor outputs
        if(!predecessor)
        {
          assert(tensor_to_node[std::make_pair(tensor, dim-1)] != nullptr);
          predecessor = tensor_to_node[std::make_pair(tensor, dim-1)];
        }
        
        
        auto tensor_access_op = rewriter.create<IndexTreeIndexToTensorOp>(
                                  loc,
                                  TypeRange({rewriter.getIndexType(), rewriter.getIndexType()}),
                                  tensor,
                                  predecessor,
                                  dim-1,
                                  nullptr);
        parent = tensor_access_op.getPos();
      }
      new_domain = rewriter.create<IndexTreeSparseDomainOp>(loc, 
                                                            domain_op->getResultTypes(),
                                                            sparse_domain_op.getTensor(),
                                                            sparse_domain_op.getDimAttr(),
                                                            sparse_domain_op.getFormatAttr(),
                                                            sparse_domain_op.getPos(),
                                                            sparse_domain_op.getCrd(),
                                                            sparse_domain_op.getPosSize(),
                                                            sparse_domain_op.getCrdSize(),
                                                            sparse_domain_op.getDimSize(),
                                                            parent);
      map.map(sparse_domain_op, new_domain);
    } else {
      // Clone
      new_domain = rewriter.clone(*domain_op, map)->getResult(0);
    }
    return new_domain;
  }

  void createMapping(IndexTreeIndicesOp node,  Value domain, llvm::SmallDenseMap<std::pair<Value, int32_t>, Value>& tensor_to_node) const
  {
    Operation* domain_op = domain.getDefiningOp();
    if(llvm::isa<IndexTreeDomainIntersectionOp>(domain_op))
    {
      auto intersection_domain_op = llvm::cast<IndexTreeDomainIntersectionOp>(domain_op);
      for(Value subdomain : intersection_domain_op.getDomains()){
        createMapping(node, subdomain, tensor_to_node);
      }
    } else if(llvm::isa<IndexTreeDomainUnionOp>(domain_op))
    {
      auto union_domain_op = llvm::cast<IndexTreeDomainUnionOp>(domain_op);
      for(Value subdomain : union_domain_op.getDomains()){
        createMapping(node, subdomain, tensor_to_node);
      }
    } else if (llvm::isa<IndexTreeNestedDomainOp>(domain_op))
    {
      auto nested_domain_op = llvm::cast<IndexTreeNestedDomainOp>(domain_op);
      for(Value subdomain : nested_domain_op.getDomains()){
        createMapping(node, subdomain, tensor_to_node);
      }
    }
    else if(llvm::isa<IndexTreeSparseDomainOp>(domain_op))
    {
      auto sparse_domain_op = llvm::cast<IndexTreeSparseDomainOp>(domain_op);
      auto tensor = sparse_domain_op.getTensor();
      int32_t dim = sparse_domain_op.getDim();
      tensor_to_node.insert(std::make_pair(
        std::make_pair(tensor, dim),
        node.getOutput()
      ));
    } else if(llvm::isa<IndexTreeDenseDomainOp>(domain_op))
    {
      auto dense_domain_op = llvm::cast<IndexTreeDenseDomainOp>(domain_op);
      auto tensors = dense_domain_op.getTensors();
      auto dims = dense_domain_op.getDims();
      unsigned i = 0;
      for(auto tensor : tensors)
      {
        int32_t dim = dims[i].cast<IntegerAttr>().getValue().getSExtValue();
        tensor_to_node.insert(std::make_pair(
          std::make_pair(tensor, dim),
          node.getOutput()
        ));
        i += 1;
      }
    }
  }

  mlir::LogicalResult
  match(IndexTreeSparseTensorOp op) const override {
    for(auto domain : op.getDomains())
    {
      Operation* domain_op = domain.getDefiningOp();
      if(domain_op->hasTrait<UnknownDomain>())
      {
        return success();
      }
    }
    return failure();
  }

  void
  rewrite(IndexTreeSparseTensorOp it_tensor_decl_op, 
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = it_tensor_decl_op->getLoc();
    auto context = rewriter.getContext();

    // Declare Sparse Domains and allocate position vectors for each dimension
    unsigned indicesBitwidth = it_tensor_decl_op->getResultTypes()[0].cast<SparseTensorType>().getIndicesType().getWidth();
    llvm::SmallDenseMap<Value, Value> symbolic_domains;

    llvm::SmallVector<Value> input_domains;

    auto domain_type = SymbolicDomainType::get(context, indicesBitwidth);
    auto index_type = rewriter.getIndexType();
    Value cur_pos_size = nullptr;
    unsigned dim = 0;
    for(Value domain : it_tensor_decl_op.getDomains()){
      Operation* domain_op = domain.getDefiningOp();

      if(domain_op->hasTrait<UnknownDomain>())
      {
        if(dim == 0)
        {
          // If this is the first dimension, the previous dimension could be expanded as "1"
          cur_pos_size = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        }

        Value num_rows = nullptr;
        BoolAttr is_dynamic = rewriter.getBoolAttr(0);
        if(cur_pos_size == nullptr){
          is_dynamic = rewriter.getBoolAttr(1);
          num_rows = rewriter.create<index::ConstantOp>(loc, index_type, rewriter.getIndexAttr(0));
        } else {
          num_rows = cur_pos_size;
        }
        auto concrete_domain = llvm::cast<indexTree::ConcreteDomain>(domain_op);
        Value dim_size = concrete_domain.getDimensionSize();
        Value symbolic_domain = rewriter.create<DeclDomainOp>(loc, domain_type, dim_size, num_rows, is_dynamic, rewriter.getI32IntegerAttr(indicesBitwidth));
        symbolic_domains.insert(std::make_pair(domain, symbolic_domain));
        input_domains.push_back(symbolic_domain);
      }
      
      if(llvm::isa<IndexTreeDenseDomainOp>(domain_op))
      {
        Value dim_size = llvm::cast<IndexTreeDenseDomainOp>(domain_op).getDimSize();
        if(dim != 0)
          cur_pos_size = rewriter.create<arith::MulIOp>(loc, rewriter.getI32Type(), cur_pos_size, dim_size);
        else
          cur_pos_size = dim_size;
      } else 
      {
        cur_pos_size = nullptr;
      }
      dim += 1;
    }

    auto itree_op = rewriter.create<IndexTreeOp>(loc, llvm::SmallVector<Type>(symbolic_domains.size(), domain_type), input_domains);
    Region* body = &itree_op.getRegion();
    loc = body->getLoc();
    Block* block = rewriter.createBlock(body);
    rewriter.setInsertionPointToStart(block);

    indexTree::IndexTreeType tree_type = indexTree::IndexTreeType::get(context);
    Value parent = rewriter.create<indexTree::IndexTreeRootOp>(loc, tree_type);
    indexTree::IndexNodeType index_node_type = indexTree::IndexNodeType::get(context);
    IRMapping map;
    llvm::SmallDenseMap<std::pair<Value, int32_t>, Value> tensor_to_node;
    SmallVector<Value> yield_args;
    Value prev_dim = parent;
    bool is_unique = true;
    for (Value domain : it_tensor_decl_op.getDomains())
    {
      Operation* domain_op = domain.getDefiningOp();
      Value prev_parent = parent;
      Value new_domain = copyDomain(domain, rewriter, loc, map, tensor_to_node, &parent);
      indexTree::IndexTreeIndicesOp index_node_op = rewriter.create<indexTree::IndexTreeIndicesOp>(loc, index_node_type, parent, new_domain);
      createMapping(index_node_op, domain, tensor_to_node);
      if(prev_parent != parent)
      {
        is_unique = false;
      }
      parent = index_node_op.getOutput();

      if(domain_op->hasTrait<UnknownDomain>())
      {
        Value symbolic_domain = symbolic_domains[domain];
        Value new_symbolic_domain = rewriter.create<indexTree::ComputeSymbolicDomainOp>(
                                      loc, 
                                      domain_type,
                                      parent,
                                      symbolic_domain,
                                      rewriter.getBoolAttr(is_unique)
        );
        new_symbolic_domain = rewriter.create<indexTree::ComputeSymbolicDomainRowOp>(
                                loc,
                                domain_type,
                                prev_dim,
                                new_symbolic_domain,
                                rewriter.getBoolAttr(!is_unique)
        );
        yield_args.push_back(new_symbolic_domain);
      }

      prev_dim = parent;
    }
    rewriter.create<indexTree::YieldOp>(loc, TypeRange(), yield_args);

    rewriter.setInsertionPointAfter(itree_op);
    SmallVector<Value, 2> args;
    unsigned i = 0;
    for (Value domain : it_tensor_decl_op.getDomains())
    {
      if(domain.getDefiningOp()->hasTrait<UnknownDomain>())
      {
        args.push_back(itree_op->getResult(i));
        i += 1;
      } else
      {
        args.push_back(domain);
      }
    }
    auto new_tensor = rewriter.create<indexTree::IndexTreeSparseTensorOp>(loc, it_tensor_decl_op->getResultTypes(), args);
    rewriter.replaceOp(it_tensor_decl_op, new_tensor->getResults());
    return;
  }
};

struct IndexTreeSymbolicComputePass : comet::impl::IndexTreeSymbolicComputePassBase<IndexTreeSymbolicComputePass> {
  using IndexTreeSymbolicComputePassBase::IndexTreeSymbolicComputePassBase;

  void runOnOperation() {
    mlir::RewritePatternSet sp_output_patterns(&getContext());
    sp_output_patterns.add<CreateSymbolicTree>(&getContext());
    mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(sp_output_patterns));
  }
};

/// Apply the compressed workspace transformations on the index tree IR
std::unique_ptr<Pass> mlir::comet::createIndexTreeSymbolicComputePass()
{
  return std::make_unique<IndexTreeSymbolicComputePass>();
}