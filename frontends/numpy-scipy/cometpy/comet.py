#
# Copyright 2022 Battelle Memorial Institute
# 
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions 
# and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
# and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
# and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

from _ast import  Attribute, BinOp, Call, Constant
import ast
import inspect
# import comet
import numpy as np
import scipy as sp
from cometpy.MLIRGen import lowering
from cometpy.MLIRGen import builders
from cometpy.MLIRGen.types import *
#import time

def get_format(A):
    if not sp.sparse.issparse(A):
        return DENSE
    elif A.format == 'csr':
        return CSR
    elif A.format == 'coo':
        return COO
    elif A.format == 'csc':
        return CSC
    else:
        raise RuntimeError('Unsupported sparse format')





class NewVisitor(ast.NodeVisitor):

    def __init__(self,inputs):
        self.no_assign = False
        self.tsemantics = {}
        self.isemantics = {}
        self.tsymbols = {"comet": None}
        self.tcurr = 0
        self.ops = []
        self.declarations = []
        self.in_args = []
        self.returns = []
        self.mask = None
        self.need_opt_comp_workspace = False
        self.dcurr = 0
        self.inputs = inputs
        self.constants = {}
        self.valueToIndex = {}
        self.currValToIndex = {}
        self.currIndexLabel = 0

        # Output formats when multiply matrices of different formats
        self.sp_matmult_conversions = {
            CSR: {
                CSR : CSR,
                COO : CSR,
                DENSE : DENSE,
            },
            COO: {
                CSR : CSR,
                COO : CSR,
                DENSE : DENSE
            },
            DENSE: {
                CSR : DENSE,
                COO : DENSE,
                DENSE : DENSE,
            }
        }

        # Output formats when transposing matrices of different formats
        # self.sp_mattr_conversions = {
        #     CSR : CSC,
        #     COO : COO,
        #     DENSE : DENSE
        # }

        self.sp_mattr_conversions = {
            CSR : CSR,
            COO : COO,
            DENSE : DENSE
        }

        # Output formats when elwise mult matrices of different formats
        # self.sp_elw_mult_conversions = {
        #     CSR: {
        #         CSR : CSR,
        #         COO : CSR,
        #         DENSE : COO,
        #     },
        #     COO: {
        #         CSR : CSR,
        #         COO : CSR,
        #         DENSE : COO
        #     },
        #     DENSE: {
        #         CSR : DENSE,
        #         COO : DENSE,
        #         DENSE : DENSE,
        #     }
        # }
        self.sp_elw_mult_conversions = {
            CSR: {
                CSR : CSR,
                COO : CSR,
                DENSE : DENSE,
            },
            COO: {
                CSR : CSR,
                COO : CSR,
                DENSE : COO
            },
            DENSE: {
                CSR : DENSE,
                COO : DENSE,
                DENSE : DENSE,
            }
        }
        # Output formats when elwise add or subtract matrices of different formats
        # Add and subtract is almost the same as elwise mult with two differences
        self.sp_elw_add_sub_conversions = self.sp_elw_mult_conversions


        # self.sp_elw_add_sub_conversions[CSR][DENSE] = DENSE
        # self.sp_elw_add_sub_conversions[COO][DENSE] = DENSE

        self.sp_elw_add_sub_conversions[CSR][DENSE] = CSR
        self.sp_elw_add_sub_conversions[COO][DENSE] = COO

    def get_next_indexlabel_with_val(self, v) :
        if v not in self.valueToIndex:
            self.valueToIndex[v] = []
            self.valueToIndex[v].append(self.currIndexLabel)
            self.currValToIndex[v] = 1
            self.currIndexLabel += 1

        else:
            if self.currValToIndex[v] >= len(self.valueToIndex[v]) :
                self.valueToIndex[v].append(self.currIndexLabel)
                self.currIndexLabel += 1

            self.currValToIndex[v] += 1
        return self.valueToIndex[v][self.currValToIndex[v]- 1]

    def reset_indexlabel_with_val(self, v) :
        self.currValToIndex[v] -= 1

    def get_index_constant(self,v) :
        if v not in self.constants:
            self.constants[v] = self.dcurr
            self.declarations.append(
                {
                    "type": "C",
                    "value": v,
                    "id": self.dcurr,
                })
            self.dcurr += 1

        return self.constants[v]

    def visit_FunctionDef(self, node):
        for i, arg in enumerate(node.args.args):
            self.tsymbols[arg.arg] = self.tcurr
            format = get_format(self.inputs[i])
            for s in self.inputs[i].shape :

                self.get_next_indexlabel_with_val(s)
                self.reset_indexlabel_with_val(s)

            self.tsemantics[self.tcurr] = {
                'shape': list(self.inputs[i].shape),
                'format': format,
                'dimsSSA': [self.get_index_constant(d) for d in self.inputs[i].shape],
                'scalar': False,
                }
            self.declarations.append(
                {
                    "type": "T",
                    "is_input": True,
                    "arg_num": i,
                    "format": format,
                    "shape": self.inputs[i].shape,
                    "dimsSSA": [self.get_index_constant(d) for d in self.inputs[i].shape],
                    "id": self.tcurr,
                })
            self.in_args.append(self.tcurr)
            self.tcurr += 1
        for stmt in node.body:
            NewVisitor.visit(self, stmt)

    
    def visit_Assign(self, node): # We do not support multiple targets currently

        if isinstance(node.targets[0], ast.Subscript): 
            id = node.targets[0].value.id
            mask = node.targets[0].slice
            self.mask = mask
            v = NewVisitor.visit(self, node.value)
            self.tsymbols[id] = v
        else:
            # vals = [NewVisitor.visit(self, node.value)]
            # for l,v in zip(node.targets, vals):
            #     self.tsymbols[l.id] = v

            id = node.targets[0].id
            # if id in self.tsymbols:
            #     self.no_assign = True

            v = NewVisitor.visit(self, node.value)
            self.tsymbols[id] = v

        

    def visit_AugAssign(self, node):
        id = node.target.id
        beta = 0
        no_assign = 0

        if isinstance(node.op, ast.Add):
            beta = 1
        elif isinstance(node.op, ast.Sub):
            beta = -1

        # if id in self.tsymbols:
        self.no_assign = beta
        no_assign = self.no_assign

        # Because comet handles +=,-= in a different way (using beta) for some operations
        # we need to specifically check for these cases.
        if isinstance(node.value, ast.Call) and node.value.func.attr == "einsum" :

            v = self.visit_Einsum_Call(node.value)
        
            if no_assign:
                self.ops.append(
                    {
                        "op_type": "=",
                        "beta": beta,
                        "lhs": self.tsymbols[id],
                        "shapes": [self.tsemantics[v]['shape'], self.tsemantics[self.tsymbols[id]]['shape']],
                        "rhs": v,
                    })
            else:
                self.tsymbols[id] = v
        elif isinstance(node.value, ast.Call) and node.value.func.attr == "transpose" :
            v = self.visit_Call(node.value)
        
            if no_assign:
                self.ops.append(
                    {
                        "op_type": "=",
                        "beta": beta,
                        "lhs": self.tsymbols[id],
                        "shapes": [self.tsemantics[v]['shape'], self.tsemantics[self.tsymbols[id]]['shape']],
                        "rhs": v,
                    })
            else:
                self.tsymbols[id] = v
        elif isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.MatMult):
            v = self.visit_BinOp(node.value)
            if no_assign:
                self.ops.append(
                    {
                        "op_type": "=",
                        "beta": beta,
                        "lhs": self.tsymbols[id],
                        "shapes": [self.tsemantics[v]['shape'], self.tsemantics[self.tsymbols[id]]['shape']],
                        "rhs": v,
                    })
            else:
                self.tsymbols[id] = v
        # Not a special case. Just add the result to the LHS
        else:
            v = NewVisitor.visit(self, node.value)
            res = self.create_binOp(node, [self.tsymbols[id], v], no_assign)
            self.tsymbols[id] = res


    def visit_Call(self, node: Call) :
        obj = None
        obj = NewVisitor.visit(self,node.func)
        if obj is not None:
            return self.visit_Method_Call(node, obj) 
        else:
            if node.func.attr == "einsum": # Handle comet.einsum
                return self.visit_Einsum_Call(node)


    def visit_Attribute(self, node: Attribute) :
        return NewVisitor.visit(self, node.value)

    def visit_Name(self, node: ast.Name):
        return self.tsymbols[node.id]

    def visit_Constant(self, node: Constant) :
        out_id = self.tcurr
        self.tsemantics[self.tcurr] = {'shape': [1,], 'format': DENSE, 'scalar': True}
        self.declarations.append(
            {
                "type": "V",
                "value": f"{node.value:e}",
                "todo": "l",
                "id": out_id,
            })
        self.tcurr +=1
        return out_id
    
    def create_binOp(self, node, operands, no_assign):
        op0_sems = self.tsemantics[operands[0]]
        op1_sems = self.tsemantics[operands[1]]
        if op0_sems['scalar'] and  op1_sems['scalar']:
            op = ''
            if isinstance(node.op, ast.Add):
                op = '+'
            elif isinstance(node.op, ast.Sub):
                op = '-'
            elif isinstance(node.op, ast.Mult):
                op = '*'
            elif isinstance(node.op, ast.Div):
                op = '/'
            else:
                raise "Unexpected operator {}".format(node.op)
            self.ops.append(
                {
                    "op_type": "scalar",
                    "op": op,
                    "operands": operands[::-1],
                    "shapes": [op1_sems['shape'], op0_sems['shape'], [1,]],
                    "out_id": self.tcurr,
                }
            )
            self.tsemantics[self.tcurr] = {'shape': [1,], 'format': DENSE, 'scalar': True}

            self.tcurr += 1
            self.declarations.append({
                "type": "V",
                "value": f"{0:e}",
                "is_input": False,
                "todo": "l",
                "format": DENSE,
                "shape": [1,],
                # "dimsSSA": [self.get_index_constant(d) for d in op_semantics['shape']],
                "id": self.tcurr,
                })
            self.ops.append(
                {
                    "op_type": "=",
                    "shapes": [[1,]]*2,
                    "lhs": self.tcurr,
                    "rhs": self.tcurr-1,
                    "beta": no_assign,
                })
            self.tsemantics[self.tcurr] = self.tsemantics[self.tcurr-1]

            self.tcurr +=1
            return self.tcurr-1

        format = self.sp_elw_add_sub_conversions[op0_sems['format']][op1_sems['format']]
        if self.tsemantics[operands[0]]['format'] != DENSE:
            op_semantics = self.tsemantics[operands[0]]
        else:
            op_semantics = self.tsemantics[operands[1]]

        self.need_opt_comp_workspace = op0_sems['format'] == CSR and op1_sems['format'] == CSR
        if isinstance(node.op, ast.Add):
            self.ops.append(
            {
                "op_type": "+",
                "shapes": [op_semantics['shape']] * 3,
                "operands": operands,
                "op_ilabels": [[self.get_next_indexlabel_with_val(d) for d in op_semantics['shape']]] * 3,
                "beta": 0,
                "out_id": self.tcurr,
            })

            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)

            self.tsemantics[self.tcurr] = {'shape': op_semantics['shape'], 'format': format, 'dimsSSA': [self.get_index_constant(d) for d in op_semantics['shape']], 'scalar': False}
            # if not no_assign:
            self.tcurr += 1
            self.declarations.append({
                "type": "T",
                "is_input": False,
                "todo": "l",
                "format": format,
                "shape": op_semantics['shape'],
                "dimsSSA": [self.get_index_constant(d) for d in op_semantics['shape']],
                "id": self.tcurr,
                })
            self.ops.append(
                {
                    "op_type": "=",
                    "shapes": [op_semantics['shape']]*2,
                    "lhs": self.tcurr,
                    "rhs": self.tcurr-1,
                    "beta": no_assign,
                })
            self.tsemantics[self.tcurr] = self.tsemantics[self.tcurr-1]
        elif isinstance(node.op, ast.Sub):
            self.ops.append(
                {
                    "op_type": "-",
                    "shapes": [op_semantics['shape']]* 3,
                    "operands": operands,
                    "op_ilabels": [[self.get_next_indexlabel_with_val(d) for d in op_semantics['shape']]] * 3,
                    "out_id": self.tcurr,
                    "beta": 0,
                })
            
            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)
            self.tsemantics[self.tcurr] = {'shape': op_semantics['shape'], 'format': format, 'dimsSSA': [self.get_index_constant(d) for d in op_semantics['shape']], 'scalar': False}
            # if not no_assign:
            self.tcurr += 1
            self.declarations.append({
                "type": "T",
                "is_input": False,
                "todo": "l",
                "format": format,
                "shape": op_semantics['shape'],
                "dimsSSA": [self.get_index_constant(d) for d in op_semantics['shape']],
                "id": self.tcurr,
                })
            self.ops.append(
                {
                    "op_type": "=",
                    "shapes": [op_semantics['shape']]*2,
                    "lhs": self.tcurr,
                    "rhs": self.tcurr-1,
                    "beta": no_assign,
                })
            self.tsemantics[self.tcurr] = self.tsemantics[self.tcurr-1]
        elif isinstance(node.op, ast.Mult):
            in_ilabels = [self.get_next_indexlabel_with_val(d) for d in op_semantics['shape']]
            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)
            
            self.ops.append(
                {
                    "op_type": "*",
                    "shapes": [op_semantics['shape']] * 3,
                    "op_ilabels": [in_ilabels] * 3,
                    "operands": operands,
                    "out_id": self.tcurr,
                    "semiring": None,
                    "beta": no_assign
                })
            format = self.sp_elw_mult_conversions[op0_sems['format']][op1_sems['format']]
            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)
            self.tsemantics[self.tcurr] = {'shape': op_semantics['shape'], 'format': format, 'dimsSSA': [self.get_index_constant(d) for d in op_semantics['shape']], 'scalar': False}
            # if not no_assign:
            self.tcurr += 1
            
            self.declarations.append({
                "type": "T",
                "is_input": False,
                "todo": "l",
                "format": format,
                "shape": op_semantics['shape'],
                "dimsSSA": [self.get_index_constant(d) for d in op_semantics['shape']],
                "id": self.tcurr,
                })
            self.ops.append(
                {
                    "op_type": "=",
                    "shapes": [op_semantics['shape']]*2,
                    "lhs": self.tcurr,
                    "rhs": self.tcurr-1,
                    "beta": no_assign,
                })
            self.tsemantics[self.tcurr] = self.tsemantics[self.tcurr-1]
        elif isinstance(node.op, ast.MatMult):
    # def visit_Bin_Einsum_Call(self, operands, llabels, mask,semiring, no_assing):

            mask = (None,None)
            if  self.mask != None:
                mask = (NewVisitor.visit(self, self.mask.elts[0]), self.mask.elts[1].value)
                self.mask = None
            op1 = self.tsemantics[operands[0]]
            op2 = self.tsemantics[operands[1]]
            if len(op1['shape']) == 2 and len(op2['shape']) == 2:
                indices = 'ij,jk->ik'
            elif len(op1['shape']) == 2 and len(op2['shape']) == 1:
                indices = 'ij,j->i'
            elif len(op1['shape']) == 1 and len(op2['shape']) == 2:
                indices = 'j,jk->k'
            
            return self.visit_Bin_Einsum_Call(operands, indices, mask, None, 0, False)

        self.tcurr +=1
        return self.tcurr-1

    def visit_BinOp(self, node: BinOp) :
        no_assign = self.no_assign
        self.no_assign = False
        operands = [NewVisitor.visit(self, node.left), NewVisitor.visit(self, node.right)]
        return self.create_binOp(node, operands, no_assign)

    def visit_Method_Call(self, node: Call, obj):
        no_assign = self.no_assign
        self.no_assign = False
        # operands = []
        out_id = self.tcurr
        op_semantics = self.tsemantics[obj]
        format = op_semantics['format']

        if node.func.attr == "transpose":
            out_format = self.sp_mattr_conversions[op_semantics['format']]
            self.tsemantics[self.tcurr] = {'shape': op_semantics['shape'][::-1], 'format': out_format, 'dimsSSA': [self.get_index_constant(d) for d in op_semantics['shape'][::-1]], 'scalar': False}
            
            in_ilabels = [self.get_next_indexlabel_with_val(d) for d in op_semantics['shape']]
            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)

            out_ilabels = [self.get_next_indexlabel_with_val(d) for d in op_semantics['shape'][::-1]]
            for d in op_semantics['shape'][::-1]:
                self.reset_indexlabel_with_val(d)

            self.ops.append(
                {
                    "op_type": "t",
                    "shapes": [op_semantics['shape'], op_semantics['shape'][::-1]],
                    "operands": [obj],
                    "op_ilabels": [in_ilabels, out_ilabels],
                    "out_id": self.tcurr,
                    "beta": 0,
                })
            self.tcurr += 1
            # if not no_assign:
            self.declarations.append(
                {
                    "type": "T",
                    "is_input": False,
                    "todo": "l",
                    "format": out_format,
                    "shape": op_semantics['shape'][::-1],
                    "dimsSSA": [self.get_index_constant(d) for d in op_semantics['shape'][::-1]],
                    "id": self.tcurr,
                })
            self.ops.append(                
                {
                    "op_type": "=",
                    "shapes": [op_semantics['shape'][::-1]]*2,
                    "lhs": self.tcurr,
                    "rhs": self.tcurr-1,
                    "beta": no_assign,
                })
            self.tsemantics[self.tcurr] = self.tsemantics[self.tcurr-1]
            
        elif node.func.attr == "sum":
            self.tsemantics[self.tcurr] = {'shape': 1, 'format': DENSE, 'scalar': True}
            self.ops.append(
                {
                    "op_type": "s",
                    "shapes": [op_semantics["shape"], [1,]],
                    "operands": [obj],
                    "out_id": self.tcurr,
                })
            # if not no_assign:
            # self.declarations.append(
            #     {
            #         "type": "V",
            #         "todo": "l",
            #         "format": DENSE,
            #         "id": self.tcurr,
            #     })
        elif node.func.attr == "multiply":
            op1 = NewVisitor.visit(self, node.args[0])
            op1_sems = self.tsemantics[op1]
            op_semantics = self.tsemantics[obj]

            in_ilabels = [self.get_next_indexlabel_with_val(d) for d in op_semantics['shape']]
            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)
            
            self.ops.append(
                {
                    "op_type": "*",
                    "shapes": [op_semantics['shape']] * 3,
                    "operands": [obj, op1],
                    "op_ilabels": [in_ilabels] * 3,
                    "out_id": self.tcurr,
                    "semiring": None,
                    "beta": no_assign,
                })

            format = self.sp_elw_mult_conversions[op_semantics['format']][op1_sems['format']]
            self.tsemantics[self.tcurr] = {'shape': op_semantics['shape'], 'format': format, 'dimsSSA': [self.get_index_constant(d) for d in op_semantics['shape'][::-1]], 'scalar': False}
            # if not no_assign:
            self.tcurr += 1
            self.declarations.append(
                {
                    "type": "T",
                    "is_input": False,
                    "todo": "l",
                    "format": format,
                    "shape": op_semantics['shape'][::-1],
                    "dimsSSA": [self.get_index_constant(d) for d in op_semantics['shape'][::-1]],
                    "id": self.tcurr,
                })
            self.ops.append(
                {
                    "op_type": "=",
                    "shapes": [op_semantics['shape'][::-1]] * 2,
                    "rhs": self.tcurr-1,
                    "lhs": self.tcurr,
                    "beta": no_assign,
                })
            self.tsemantics[self.tcurr] = self.tsemantics[self.tcurr-1]

        self.tcurr +=1

        return self.tcurr-1

    def visit_Return(self, node):
        obj = NewVisitor.visit(self, node.value)
        self.returns.append(obj)
        if self.tsemantics[obj]['format'] == DENSE:
            self.in_args.append(obj)
        self.ops.append(
            {
                "op_type": "p",
                "shapes": [self.tsemantics[obj]["shape"]],
                "value_type": "f64",
                "operands": [obj]
            })

    def visit_Bin_Einsum_Call(self, operands, llabels, mask,semiring, beta, no_assign):

        ops = llabels.split('->')[0].split(',')
        res = list(llabels.split('->')[1])
        # ret_num = []
        all_real_lbls = []
        all_dims = self.tsemantics[operands[0]]['shape'][:] # copy
        all_lbls = list(ops[0])
        if len(ops) > 1:
            for i,l in enumerate(list(ops[1])):
                if l not in all_lbls:
                    all_lbls.append(l)
                    all_dims.append(self.tsemantics[operands[1]]['shape'][i])
        
        lbls_to_dims = {}
        lbls_to_ilbls = {}
        ilbls_seen = set()

        for d,l in zip(all_dims, all_lbls):
            lbls_to_dims[l] = d
            lbls_to_ilbls[l] = self.get_next_indexlabel_with_val(d)
        
        for d,l in zip(all_dims, all_lbls):
            self.reset_indexlabel_with_val(d)


        ret_num = [i for x in res for  i,v in enumerate(all_lbls) if x == v]
        in0_ilabels = [lbls_to_ilbls[l] for l in list(ops[0])]
        in1_ilabels = []
        if len(ops) > 1:
            in1_ilabels = [lbls_to_ilbls[l] for l in list(ops[1])]
        
        format = self.tsemantics[operands[0]]['format']
        if format != DENSE:
                self.need_opt_comp_workspace = True

        all_dims = self.tsemantics[operands[0]]['shape'][:]
        all_lbls = in0_ilabels[:]
        if len(ops) > 1:
            for l,d in zip(in1_ilabels,self.tsemantics[operands[1]]['shape']):
                if l not in all_lbls:
                    all_lbls.append(l)
                    all_dims.append(d)


        shape = [all_dims[v] for v in ret_num]
        labels = [all_lbls[v] for v in ret_num]
        if len(ops) > 1:
            for op in operands[1:]:
                format = self.sp_matmult_conversions[format][self.tsemantics[op]['format']]
                if format != DENSE:
                    self.need_opt_comp_workspace = True

            self.ops.append(
                {
                    "op_type": "c",
                    "operands": operands,
                    "op_ilabels": [in0_ilabels, in1_ilabels, labels],
                    "shapes": [self.tsemantics[operands[0]]['shape'], self.tsemantics[operands[1]]['shape'], shape],
                    "out_id": self.tcurr,
                    "mask": mask,
                    "semiring": semiring,
                    "beta": beta
                })
                
        elif len(ops) == 1:
            in_ilabels = [self.get_next_indexlabel_with_val(d) for d in self.tsemantics[operands[0]]['shape']]
            for d in self.tsemantics[operands[0]]['shape']:
                self.reset_indexlabel_with_val(d)
            
            out_ilabels = [self.get_next_indexlabel_with_val(d) for d in shape]
            for d in shape:
                self.reset_indexlabel_with_val(d)
            
            self.ops.append(
                {
                    "op_type": "t",
                    "shapes": [self.tsemantics[operands[0]]['shape'], shape],
                    "operands": operands,
                    "out_id": self.tcurr,
                    "beta": beta,
                    "op_ilabels": [in_ilabels, out_ilabels]
                })
                
        self.tsemantics[self.tcurr] = {'shape': shape, 'format': format, 'dimsSSA': [self.get_index_constant(d) for d in shape], 'scalar': False}
        if not no_assign:
            self.tcurr += 1
            self.declarations.append(
                {
                    "type": "T",
                    "is_input": False,
                    "todo": "l",
                    "format": format,
                    "shape": shape,
                    "dimsSSA": [self.get_index_constant(d) for d in shape],
                    "id": self.tcurr,
                })
            self.ops.append(                
                {
                    "op_type": "=",
                    "shapes": [shape] * 2,
                    "lhs": self.tcurr,
                    "rhs": self.tcurr-1,
                    "beta": beta,
                })
            self.tsemantics[self.tcurr] = self.tsemantics[self.tcurr-1]
        
        self.tcurr += 1

        return  self.tcurr - 1

    def visit_Einsum_Call(self, node: Call):
        no_assign =  self.no_assign
        self.no_assign = False
        out_id = self.tcurr
        mask = (None, None)
        # mask_type = "none"
        semiring = None
        if  self.mask != None:
            mask = (NewVisitor.visit(self, self.mask.elts[0]), self.mask.elts[1].value)

            self.mask = None
        iLabels = node.args[0].value
        ops, res = iLabels.split('->')
        ops = ops.split(',')
        res = list(res)
        if len(node.keywords) > 0:
            for key in node.keywords:
                if key.arg == 'mask':
                    mask = NewVisitor.visit(self, key.value)

                    mask = (mask, "pull")
                elif key.arg == 'semiring':
                    semiring = key.value.value
                elif key.arg == 'mask_type':
                    mask = (mask[0], key.value.value)

        operands = []
        if len(node.args) > 1:
            for arg in node.args[1:]:
                operands.append(NewVisitor.visit(self, arg))
        if semiring != None and len(semiring.split(',')) == 1:
            op0_sems = self.tsemantics[operands[0]]
            op1_sems = self.tsemantics[operands[1]]
            

            format = self.sp_elw_add_sub_conversions[op0_sems['format']][op1_sems['format']]
            if self.tsemantics[operands[0]]['format'] != DENSE:
                op_semantics = self.tsemantics[operands[0]]
            else:
                op_semantics = self.tsemantics[operands[1]]

            in_ilabels = [self.get_next_indexlabel_with_val(d) for d in op_semantics['shape']]
            for d in op_semantics['shape']:
                self.reset_indexlabel_with_val(d)
            
            self.ops.append(
                {
                    "op_type": "*",
                    "shapes": [op_semantics['shape']] * 3,
                    "operands": operands,
                    "op_ilabels": [in_ilabels] * 3,
                    "out_id": self.tcurr,
                    "semiring": semiring,
                    "beta": no_assign
                })
                
                # ("*", operands, indices+','+indices+'->'+indices, self.tcurr, semiring))
            format = self.sp_elw_mult_conversions[op0_sems['format']][op1_sems['format']]
            self.tsemantics[self.tcurr] = {'shape': op_semantics['shape'], 'format': format, 'dimsSSA': [self.get_index_constant(d) for d in op_semantics['shape']], 'scalar': False }
            self.tcurr += 1
            self.declarations.append(
                {
                    "type": "T",
                    "is_input": False,
                    "todo": "l",
                    "format": format,
                    "shape": op_semantics['shape'],
                    "dimsSSA": [self.get_index_constant(d) for d in op_semantics['shape']],
                    "id": self.tcurr,
                })
            self.ops.append(
                {
                    "op_type": "=",
                    "shapes": [op_semantics['shape']] * 2,
                    "rhs": self.tcurr-1,
                    "lhs": self.tcurr,
                    "beta": no_assign,
                })
            self.tsemantics[self.tcurr] = self.tsemantics[self.tcurr-1]
            self.tcurr += 1

            return self.tcurr-1
        if len(operands) > 1: # Contraction
            loperand = operands[0]
            lops = ops[0]
            saved_no_assign = no_assign
            no_assign = True
            for i in range(1, len(operands)):
                all_lbls = list(lops)
                for l in list(ops[i]):
                    if l not in all_lbls:
                        all_lbls.append(l)

                rh1_lbls_num = [curr for x in list(lops) for  curr,v in enumerate(all_lbls) if x == v]
                rh2_lbls_num = [curr for x in list(ops[i]) for  curr,v in enumerate(all_lbls) if x == v]
                all_lbls_num = rh1_lbls_num[:] # Copy
                for n in rh2_lbls_num:
                    if n not in all_lbls_num:
                        all_lbls_num.append(n)

                sum = set(rh1_lbls_num) & set(rh2_lbls_num) 
                all = set(rh1_lbls_num) | set(rh2_lbls_num) 
                ret = all - sum

                ret_num = [i for x in ret for  i,v in enumerate(all_lbls_num) if x == v]
                ret = [all_lbls[x] for x in ret_num]
                    



                if i == len(operands) -1 :
                    ret = res
                    no_assign = saved_no_assign
                tid = self.visit_Bin_Einsum_Call([loperand,operands[i]], lops+","+ops[i]+"->"+"".join(ret), mask, semiring, saved_no_assign, no_assign)
                loperand = tid
                lops = "".join(ret)
            return tid
        elif len(operands) == 1: # transpose

            loperand = operands[0]
            lops = ops[0]
            ret = res
            tid = self.visit_Bin_Einsum_Call([loperand], lops+"->"+"".join(ret), mask, semiring, no_assign, no_assign)
            lops = "".join(ret)
            return tid


#Wrapper function. The input function (in the form of an object) is passed an arguement to this function.
def compile(flags, target:str = "cpu", with_jit=True):
    def innerfunc(func):

        def wrapper(*pos_args, **kwargs):
            func_str = ast.parse(inspect.getsource(func))
            parsed_func = ast.parse(func_str)
            func_def = parsed_func.body[0]
            v = NewVisitor([*pos_args])
            v.visit(parsed_func)
            in_types = []
            for arg in v.in_args:
                if isinstance(v.tsemantics[arg]['shape'], int):
                    in_types.append(("%t"+str(arg), "tensor<1xf64>"))
                else:
                    in_types.append(("%t"+str(arg), "tensor<{}xf64>".format("x".join(str(d) for d in v.tsemantics[arg]['shape']))))
            irb = builders.MLIRFunctionBuilder(
                func_def.name,
                input_types=in_types,
                return_types=[],
            ) 

            for i in range(v.currIndexLabel):
                irb.add_statement('%i{} = "ta.index_label"() : () -> !ta.indexlabel'.format(i))


            dense_tensors = []
            scalars = []
            for dec in v.declarations:
                
                if dec["type"] == "T":
                    dec["value_type"] = "f64"
                    t = builders.Tensor_Decl_Builder(dec)
                    if dec["format"] == DENSE:
                        dense_tensors.append(t)
                    else:
                        irb.add_statement(t.build_tensor())
                elif dec["type"] == "C":
                    irb.add_statement('%d{} = arith.constant {} : index '.format(dec["id"], dec["value"]))
                elif dec["type"] == "V":
                    scalars.append('%t{} = ta.constant dense<{}> : tensor<1xf64> '.format(dec["id"], dec["value"]))
                    # irb.add_statement('%t{} = ta.constant dense<{}> : tensor<1xf64> '.format(dec["id"], dec["value"]))


            for t in dense_tensors:
                irb.add_statement(t.build_tensor())

            for t in scalars:
                irb.add_statement(t)

            for op in v.ops:
                if op["op_type"] == 'c':
                    op["formats"] = [v.tsemantics[t]['format'] for t in op["operands"]] +  [v.tsemantics[op["out_id"]]['format']]
                    if op["mask"][0] is not None :
                        op["mask"] = (op["mask"][0], op["mask"][1], v.tsemantics[op["mask"][0]]['shape'])
                        irb.add_statement(builders.ArithOp_Builder(op).build_op()) 
                    else:
                        op["mask"] = (op["mask"][0], op["mask"][1], None)
                        irb.add_statement(builders.ArithOp_Builder(op).build_op()) 
                elif op["op_type"] == 'scalar':
                    irb.add_statement(builders.ScalarOp_Builder(op).build_op())
                elif op["op_type"] == 's':
                    irb.add_statement(builders.TensorSumBuilder(op).build_op())
                elif op["op_type"] == 'p':
                    irb.add_statement(builders.PrintBuilder(op).build_op())
                elif op["op_type"] == '*':
                    op["formats"] = [v.tsemantics[t]['format'] for t in op["operands"]] +  [v.tsemantics[op["out_id"]]['format']]
                    irb.add_statement(builders.ArithOp_Builder(op).build_op())
                elif op["op_type"] == '=':
                    irb.add_statement(builders.SetOp_Builder(op).build_op())
                else:
                    op["formats"] = [v.tsemantics[t]['format'] for t in op["operands"]] +  [v.tsemantics[op["out_id"]]['format']]
                    irb.add_statement(builders.ArithOp_Builder(op).build_op()) 
            irb.add_statement("return")

            outputs = []
            ret = v.tsemantics[v.returns[0]]
            format = ret['format']
            if format == DENSE:
                outputs.append(np.empty(ret['shape']))
            elif format == CSR:
                outputs.append(sp.sparse.csr_array(np.empty(ret['shape'])))
            elif format == COO:
                outputs.append(sp.sparse.coo_array(np.empty(ret['shape'])))
            elif format == CSC:
                outputs.append(sp.sparse.csc_array(np.empty(ret['shape'])))

            arg_vals = v.inputs
            new_flags = flags
            if v.need_opt_comp_workspace:
                if new_flags:
                    new_flags = new_flags + ' --opt-comp-workspace'
                else:
                    new_flags = ' --opt-comp-workspace'
            code = irb.compile()
            # start = time.time()
            lowering_result = lowering.lower_dialect_with_jit(code, target, None, new_flags,func_def.name, arg_vals, outputs)
            # end = time.time()
            # print("Time for JIT", end-start)
            return lowering_result

        return wrapper
    return innerfunc
