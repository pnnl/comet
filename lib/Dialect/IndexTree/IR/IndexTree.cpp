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

#include <stack>
#include <thread>
#include "comet/Dialect/IndexTree/IR/IndexTree.h"
#include "comet/Dialect/IndexTree/Transforms/UnitExpression.h"

using namespace std;

// *********** For debug purpose *********//
// #define COMET_DEBUG_MODE
#include "comet/Utils/debug.h"
#undef COMET_DEBUG_MODE
// *********** For debug purpose *********//

// *********** For debug purpose *********//

string TreeNode::getName()
{
  if (index == -1)
  {
    return "-1";
  }
  else
  {
    return Index_Tree::getIndexName(index);
  }
}

void TreeNode::print(int depth)
{
  for (int i = 0; i < depth; i++)
  {
    comet_debug() << "  ";
  }
  if (isRealIndexNode())
  {
    if (outputDomain != nullptr)
    {
      comet_debug() << "for " << getName() << " in input " << inputDomain->str() << ", output " << outputDomain->str() << '\n';
    }
    else
    {
      comet_debug() << "for " << getName() << " in input " << inputDomain->str() << '\n';
    }
    for (auto &child : children)
    {
      child->print(depth + 1);
    }
  }
  else if (isFillerIndexNode())
  {
    for (auto &child : children)
    {
      child->print(depth);
    }
  }
  else
  {
    assert(expr != nullptr);
    comet_debug() << expr->str() << '\n';
  }
}

int TreeNode::getId() const
{
  return id;
}

void TreeNode::setId(int Id)
{
  id = Id;
}

void Index_Tree::print(string msg)
{
  comet_debug() << msg << "\n";
  getRoot()->print(0);
}

IndicesType Index_Tree::getIndices(std::vector<mlir::Value> &lbls)
{
  IndicesType indices;

  for (auto lbl : lbls)
  {
    comet_vdump(lbl);
    void *lbl_ptr = lbl.getAsOpaquePointer();
    if (indexLabelToId.count(lbl_ptr) == 0)
    {
      comet_debug() << "Index Label just created:" << indexID << "\n";
      indexLabelToId[lbl_ptr] = indexID;
      indexID++;
    }
    indices.push_back(indexLabelToId[lbl_ptr]);
  }

  return indices;
}

Tensor *Index_Tree::getOrCreateTensor(mlir::Value v, std::vector<mlir::Value> &allIndexLabels, FormatsType &formats)
{
  IndicesType indices = getIndices(allIndexLabels);
  comet_debug() << "Num Indices: " << indices.size() << ", Num formats " << formats.size() << "\n";

  return new Tensor(v, indices, formats);
}

TreeNode *Index_Tree::addComputeNode(unique_ptr<UnitExpression> expr, TreeNode *parent)
{
  assert(parent != nullptr);
  assert(parent->isRealIndexNode());
  auto node = make_unique<TreeNode>(expr.get(), parent);
  auto nodePtr = registerNode(node);
  parent->appendChild(nodePtr);
  exprToNode[expr.get()] = nodePtr;
  expressions.push_back(std::move(expr));
  return nodePtr;
}

unique_ptr<Index_Tree> Index_Tree::createTreeWithRoot()
{
  auto tree = make_unique<Index_Tree>();
  tree->addRootNode();
  return tree;
}

void visitExpr(UnitExpression *expr, set<UnitExpression *> &visited,
               vector<UnitExpression *> &results)
{
  if (visited.count(expr) > 0)
  {
    return;
  }

  visited.insert(expr);

  for (auto user : expr->getUsers())
  {
    visitExpr(user, visited, results);
  }

  results.push_back(expr);
}

vector<UnitExpression *> Index_Tree::getExpressionsInTopoOrder()
{
  vector<UnitExpression *> results;
  set<UnitExpression *> visited;
  for (auto e : getExpressions())
  {
    visitExpr(e, visited, results);
  }

  std::reverse(results.begin(), results.end());
  return results;
}

vector<UnitExpression *> Index_Tree::getContainingExpressions(TreeNode *node)
{
  vector<UnitExpression *> exprs;
  stack<TreeNode *> nodes;
  nodes.push(node);
  while (!nodes.empty())
  {
    auto cur = nodes.top();
    nodes.pop();

    if (cur->isComputeNode())
    {
      exprs.push_back(cur->getExpression());
    }
    else
    {
      for (auto child : cur->getChildren())
      {
        nodes.push(child);
      }
    }
  }
  return exprs;
}

vector<TreeNode *> Index_Tree::getNodes()
{
  vector<TreeNode *> nodePtrs;
  for (auto &n : nodes)
  {
    nodePtrs.push_back(n.get());
  }
  return nodePtrs;
}

void visitNode(TreeNode *node, set<TreeNode *> &visited, vector<TreeNode *> &results)
{
  if (visited.count(node) > 0)
  {
    return;
  }
  visited.insert(node);

  for (auto c : node->getChildren())
  {
    visitNode(c, visited, results);
  }

  results.push_back(node);
}

vector<TreeNode *> Index_Tree::getNodesInReverseTopoOrder()
{
  vector<TreeNode *> results;
  set<TreeNode *> visited;
  for (auto &n : getNodes())
  {
    visitNode(n, visited, results);
  }

  // std::reverse(results.begin(), results.end());
  return results;
}

vector<mlir::Operation *> Index_Tree::getContainingTAOps()
{
  vector<mlir::Operation *> ops;
  for (auto &e : getExpressions())
  {
    if (e->getOperation())
      ops.push_back(e->getOperation());
  }
  return ops;
}

std::unordered_set<std::string> IteratorType::supported_types = {"default",
                                                                 "serial",
                                                                 "parallel",
                                                                 "omp.parallel",
                                                                 "reduction",
                                                                 "window"};
void IteratorType::setType(std::string t) {
  if (supported_types.find(t) != supported_types.end()) {
    type = t;
  } else {
    llvm::errs() << "Unsupported iterator type " + t + "\n";
  }
}