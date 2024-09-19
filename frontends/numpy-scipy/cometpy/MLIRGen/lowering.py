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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import os
import subprocess
import shlex
import numpy as np
import platform
import sys
import time
import ctypes
from ctypes import *
import cometpy.cfg as cfg
import atexit
import uuid
import scipy as scp
debug = False
temp_dir = '.cometpy/'

if not os.path.exists(temp_dir):
    try:
        os.mkdir(temp_dir)
    except:
        if os.path.exists(temp_dir):
            pass
        else:
            raise "Could not create .cometpy/"

platform_args = ""


if("macOS" in platform.platform()):
    comet_runner_util = cfg.comet_path+"/lib/libcomet_runner_utils.dylib"
    platform_args = "-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/ "
elif("Linux" in platform.platform()):
    comet_runner_util = cfg.comet_path+"/lib/libcomet_runner_utils.so"
else:
    print("error: Support available only for Linux and macOS")
    sys.exit()
files_to_cleanup = []
def cleanup():
    for f in files_to_cleanup:
        if os.path.exists(f):
            os.remove(f)
            # pass
atexit.register(cleanup)

class memref_i64(Structure):
    _fields_ = [ ('mem_aligned', POINTER(c_longlong)), ('mem', POINTER(c_longlong)), ('offset', c_longlong), ('dim', c_longlong), ('stride', c_longlong)]

class memref_f64(Structure):
    _fields_ = [ ('mem_aligned', POINTER(c_double)), ('mem', POINTER(c_double)), ('offset', c_longlong), ('dim', c_longlong), ('stride', c_longlong)]


def np_array_to_memref(np_array):
    ctype = ctypes.c_longlong
    if np_array.dtype == 'int32':
        ctype = c_int32
    elif np_array.dtype == 'float32':
        ctype = c_float
    elif np_array.dtype == 'float64':
        ctype = c_double
    return np_array.ctypes.data_as(ctypes.POINTER(ctype)), np_array.ctypes.data_as(ctypes.POINTER(ctype)), 0, np_array.shape[0], 1

def expand_memref_ptr(memref):
    return byref(memref), byref(memref), 0, 1, 1

def len_dense(vals):
    num = 0
    for v in vals:
        if not scp.sparse.issparse(v):
            num+=1
    return num




def lower_ta_to_mlir(mlir_in, mlir_lower_flags, uuid_s):

    # scf_out_file = 'einsum_loops.mlir'
    scf_out_file = uuid_s+'loops.mlir'

    if(os.path.exists(scf_out_file) == False):
        f = open(os.path.join( os.getcwd(), scf_out_file), 'wb')
        files_to_cleanup.append(os.path.join( os.getcwd(), scf_out_file))
    else:
        f = open(scf_out_file, 'wb')

    path_to_comet = cfg.comet_path+"/bin/comet-opt"

    command = path_to_comet + mlir_lower_flags + mlir_in

    p = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,close_fds=False)
    if(p.returncode != 0):
        cleanup()
        raise AssertionError("comet-opt failed with error code: {}. Error message: {}".format(p.returncode, p.stderr.decode()))
    
    scf_out  = p.stderr
    f.write(scf_out)
    f.close()

    return scf_out_file

def all_dense(arg_vals) -> bool :
    for v in arg_vals:
        if scp.sparse.issparse(v):
            return False;
    return True

def comment_unneeded_dense(input_, arg_vals):
    input = input_.splitlines()
    outs = []
    indexes = []
    for i, v in enumerate(arg_vals):
        if not scp.sparse.issparse(v):
            indexes.append(i)
    replace = {}
    fill_remove = []

    allocs_needed = len_dense(arg_vals)

    for i in range(len(input)):
        if "call @comet_print_memref_f64" in input[i]:
            cast = input[i][input[i].find("(") + 1 :  input[i].find(")")]
            # input[i]  = "// from dense " + input[i]
            input[i]  = ""
            for j in range(len(input[:i])):
                if cast + " = memref.cast" in input[j]:
                    out = input[j][input[j].find("%alloc")  : input[j].find(":")].lstrip().strip()
                    outs.append(out)
                    start = input[j].find(":")
                    end = input[j][start:].find("to")
                    replace['%arg'+str(len(arg_vals))] = out +" " + input[j][start:start+end].lstrip().strip()
                    for k in range(len(input[:j])):
                        if out+" = memref.alloc(" in  input[k]:
                            # input[k] = "//from dense" + input[k]
                            input[k] = ""
        elif allocs_needed > 0 and "memref.alloc" in input[i]:
            allocs_needed = allocs_needed - 1
            a = input[i][input[i].find("%") : input[i].find("=")].lstrip().strip()
            start = input[i].rfind(":")
            replace['%arg'+str(indexes[0])] = a +" " + input[i][start:].lstrip().strip()
            indexes = indexes[1:]
            # input[i] = "//from dense" + input[i]
            input[i] = ""
            fill_remove.append(a)

        elif "linalg.fill" in input[i]:
            for k in range(len(fill_remove)):
                if fill_remove[k] in input[i]:
                    # input[i] = "//from dense" + input[i]
                    input[i] = ""
                    fill_remove.remove(fill_remove[k])
                    break

    for v in replace:
        start = input[1].find(v)
        end = input[1][start:].find(",")
        if end == -1 :
            end = input[1][start:].find(")")
        repl = input[1][start:start+end]
        input[1] = input[1].replace(repl, replace[v])
            
    output = ""
    arg_vals_init = {}
    input_arg_names = []
    collected = False
    for i in range(len(input)):
        l = input[i]
        if not collected and l.lstrip().strip().startswith("func.func @") and l.strip().endswith("{"):
            for arg in l[l.find("(")+1:l.find(")")].split(","):
                if arg.split(":")[0].strip().lstrip() not in outs:
                    input_arg_names.append(arg.split(":")[0].strip().lstrip())
            for v in input_arg_names:
                arg_vals_init[v] = False
            collected = True
        elif "memref.store" in input[i] :
            for arg in input_arg_names:
                if arg+"[" in input[i] and arg_vals_init[arg] == False:
                    init = input[i].lstrip().split(" ")[1][:-1].lstrip().strip()
                    for l in input:
                        if l.lstrip().strip().startswith(init) and "arith.constant" in l:
                            if float(l.strip().lstrip().split(" ")[3]) == 0.0 :
                                input[i] = ""
                                arg_vals_init[arg] = True
                                break


    for line in input:
        if line:
            output += line +"\n"

    return output            

def comment_unneeded_sparse(input_, arg_vals):
    output = ""
    input = input_.splitlines()
    indexes = []
    indexes = []
    allocs = []
    alloc_lines = []
    returns = []
    return_found = False
    out_len = len(input)
    for i in range(len(input)):
        line = input[i]

        if "call @read_input_sizes" in input[i]:
            
            # cast = line[line.find("(") : line.find(")")].split(",")[3].lstrip().strip()
            # With tiles
            cast = line[line.find("(") : line.find(")")].split(",")[5].lstrip().strip()
            
            input[i] = "//from sparse" +input[i]
            # input[i] = ""
            alloc = ""
            for j in range(len(input[:i])):
                lline = input[j]
                if cast +" = memref.cast" in lline:
                    alloc = lline.split()[3].lstrip().strip()
                    # input[j] = "// from sparse" + input[j]
                    # input[j] = ""
                    for k in range(len(input[:j])):
                        if alloc + " = memref.alloc" in input[k]:
                            # input[k] = "// from sparse"+input[k]
                            input[k] = ""
                            allocs.append(alloc)
            i+=1
            found = 0

            # while(found != 7):
            # With tiles
            while(found != 9):
                if "memref.load " + alloc in input[i]:
                    idx = input[i].split('=')[0].lstrip().strip()
                    indexes.append(idx)
                    found += 1
                i+=1
            idx = 0
        elif 'memref.alloc(' in input[i]:
            line = input[i]
            idx = line[line.find('(') + 1: line.find(')')]
            # if idx in indexes[:-1]:
            if idx in indexes:
                indexes.remove(idx)
                allocs.append(line.split('=')[0].rstrip().strip())
                cur_lines = []
                cur_lines.append(i)
                while "scf.for" not in input[i]:
                    i+=1
                cur_lines.append(i)
                cur_lines.append(i+1)
                cur_lines.append(i+2)
                cur_lines.append(i+3)
                alloc_lines.append(cur_lines)
        elif "call @read_input_2D" in input[i]:
            # casts = input[i].split(",")[3:8]
            # With tiles
            casts = input[i].split(",")[5:14]
            for lines in alloc_lines:
                if "memref.cast" in input[lines[-1]]:
                    for c in casts:
                        if c in input[lines[-1]]:
                            for l in lines[:-1]:
                                # input[l] = "// from sparse new " + input[l]
                                input[l] = ""
            input[i] = '// from sparse' + input[i]
            # input[i] = ""
        elif "call @comet_print_memref_i64" in input[i] or "call @comet_print_memref_f64" in input[i]:
            cast = input[i][input[i].find("(") + 1 : input[i].find(")")]
            for j in range(len(input[:i])):
                lline = input[j]
                if cast + " = memref.cast" in lline:
                    alloc = lline.split()[3].lstrip().strip()
                    type = lline.split(":")[1].split("to")[0].strip()
                    returns.append((alloc, i, type))
        elif ("return" in input[i]) and len(returns) > 1 and not return_found:
            return_found = True
            add = ""
            for k, r in enumerate(returns[:-1]):
                add += "\t\tmemref.store {}, %marg{}[%c0] : memref<1x{}>\n".format(r[0], k, r[2] )
                # input[r[1]] = "//from sparse" + input[r[1]]
                input[r[1]] = ""
            # input[returns[-1][1]] = "//from sparse" + input[returns[-1][1]]
            input[returns[-1][1]] = ""
            add += "\t\tmemref.store {}, %marg{}[%c0] : memref<1x{}>\n".format(returns[-1][0], len(returns)-1, returns[-1][2])
            add += "\t\treturn"
            input[i] = add

    args = input[1][input[1].find("(") + 1: input[1].find(")")].split(",")
    ai = 0
    for i, v in enumerate(arg_vals):
        if scp.sparse.issparse(v):
            
            # input[1] = input[1].replace(args[i], allocs[ai] +" : memref<7xindex>, " + " : memref<?xindex>, ".join([s for s in allocs[ai+1:ai+6]]) + " : memref<?xf64>")
            # ai += 6
            # With tiles
            input[1] = input[1].replace(args[i], allocs[ai] +" : memref<13xindex>, " + " : memref<?xindex>, ".join([s for s in allocs[ai+1:ai+10]]) + " : memref<?xf64>")
            ai += 10 
    
    if len(returns) > 1:
#         input[1] = input[1].replace(")", ", %marg0: memref<1xmemref<?xindex>>, %marg1: memref<1xmemref<?xindex>>, %marg2: memref<1xmemref<?xindex>>, %marg3: memref<1xmemref<?xindex>>, %marg4: memref<1xmemref<?xf64>>)")
#     input[-1] = '\n  func.func @dealloc(%to_dealloc: memref<?xindex>, %to_dealloc1: memref<?xindex>, %to_dealloc2: memref<?xindex>, %to_dealloc3: memref<?xindex>, %to_dealloc4: memref<?xf64>){\n \
# \t\tmemref.dealloc %to_dealloc : memref<?xindex>\n \
# \t\tmemref.dealloc %to_dealloc1 : memref<?xindex>\n \
# \t\tmemref.dealloc %to_dealloc2 : memref<?xindex>\n \
# \t\tmemref.dealloc %to_dealloc3 : memref<?xindex>\n \
# \t\tmemref.dealloc %to_dealloc4 : memref<?xf64>\n \
# \t\treturn\n \
# \t}\n\
# }\n'
    # With tiles
            input[1] = input[1].replace(")", ", %marg0: memref<1x{}>, %marg1: memref<1x{}>, %marg2: memref<1x{}>, %marg3: memref<1x{}>, %marg4: memref<1x{}>, %marg5: memref<1x{}>, %marg6: memref<1x{}>, %marg7: memref<1x{}>, %marg8: memref<1x{}>)".format(*[x[2] for x in returns]))
    input[-1] = '\n  func.func @dealloc(%to_dealloc: memref<?xindex>, %to_dealloc1: memref<?xindex>, %to_dealloc2: memref<?xindex>, %to_dealloc3: memref<?xindex>, %to_dealloc4: memref<?xindex>, %to_dealloc5: memref<?xindex>, %to_dealloc6: memref<?xindex>, %to_dealloc7: memref<?xindex>, %to_dealloc8: memref<?xf64>){\n \
\t\tmemref.dealloc %to_dealloc : memref<?xindex>\n \
\t\tmemref.dealloc %to_dealloc1 : memref<?xindex>\n \
\t\tmemref.dealloc %to_dealloc2 : memref<?xindex>\n \
\t\tmemref.dealloc %to_dealloc3 : memref<?xindex>\n \
\t\tmemref.dealloc %to_dealloc4 : memref<?xindex>\n \
\t\tmemref.dealloc %to_dealloc5 : memref<?xindex>\n \
\t\tmemref.dealloc %to_dealloc6 : memref<?xindex>\n \
\t\tmemref.dealloc %to_dealloc7 : memref<?xindex>\n \
\t\tmemref.dealloc %to_dealloc8 : memref<?xf64>\n \
\t\treturn\n \
\t}\n\
}\n'
    for line in input:
        if line:
            output += line +"\n"
    return output

def lower_ta_to_mlir_with_jit(mlir_in, mlir_lower_flags, arg_vals, uuid_s):

    path_to_comet = cfg.comet_path+"/bin/comet-opt -x mlir "
    command = path_to_comet + mlir_lower_flags
    p = subprocess.run(shlex.split(command), input = mlir_in.encode('utf-8') , stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,close_fds=False)
    if p.returncode != 0:
        cleanup()
        raise AssertionError("comet-opt failed with error code: {}. Error: {}".format(p.returncode, p.stderr.decode()))


    # scf_out_file = temp_dir + uuid_s+'loops.mlir'

    # if(os.path.exists(scf_out_file) == False):
    #     f = open(os.path.join( os.getcwd(), scf_out_file), 'w')
    #     files_to_cleanup.append(os.path.join( os.getcwd(), scf_out_file))
    # else:
    #     f = open(scf_out_file, 'w')
    #     files_to_cleanup.append(os.path.join( os.getcwd(), scf_out_file))

    scf_out  = p.stderr.decode()

    scf_out = comment_unneeded_sparse(scf_out, arg_vals)
    scf_out = comment_unneeded_dense(scf_out, arg_vals)
    # f.write(scf_out)
    # f.close()

    return scf_out


def lower_scf_to_llvm(scf_in, scf_lower_flags, uuid_s):

    # llvm_out_file = 'einsum.llvm'

    # path_to_mliropt = "../llvm/build/bin/mlir-opt"
    path_to_cometopt = cfg.comet_path+"/bin/comet-opt"

    command = path_to_cometopt + scf_lower_flags + scf_in
    # command = path_to_cometopt + scf_lower_flags + ' 38ee2ea1-6e20-439f-b7c0-73d7fa6b7da5loops.mlir'

    p = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)
    if(p.returncode != 0):
        cleanup()
        raise AssertionError("comet-opt failed with error code: {}. Error message: {}".format(p.returncode, p.stderr.decode()))

    llvm_out_file = temp_dir + uuid_s+'.llvm'

    if(os.path.exists(llvm_out_file) == False):
        f = open(os.path.join( os.getcwd(), llvm_out_file), 'wb')
        files_to_cleanup.append(os.path.join( os.getcwd(), llvm_out_file))
    else:
        f = open(llvm_out_file, 'wb')

    llvm_out = p.stderr

    f.write(llvm_out)
    f.close()

    return llvm_out_file

# def execute_llvm(llvm_in,func_name):
    
#     path_to_mlir_cpu_runner = "../llvm/build/bin/mlir-cpu-runner "

#     flags = ' -O3 -e ' + func_name + ' -entry-point-result=void -shared-libs=../llvm/build/lib/libmlir_runner_utils.dylib,../build/lib/libcomet_runner_utils.dylib'

#     command = path_to_mlir_cpu_runner + llvm_in + flags

#     p = subprocess.run(shlex.split(command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)

#     result = p.stdout
        
#     return result

def generate_llvm_args_from_ndarrays(num_in, *ndargs):
    llvm_args = []
    llvm_args_types = []
    all_outputs = []
    for i, ndarray in enumerate(ndargs):
        # Ndarray is dense
        if not scp.sparse.issparse(ndarray):
            llvm_args.append(ndarray.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            llvm_args.append(ndarray.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            llvm_args.append(0)
            llvm_args_types.append(ctypes.POINTER(ctypes.c_double))
            llvm_args_types.append(ctypes.POINTER(ctypes.c_double))
            llvm_args_types.append(ctypes.c_longlong)
            for s in ndarray.shape:
                llvm_args.append(s)
                llvm_args_types.append(ctypes.c_longlong)

            for s in ndarray.strides:
                llvm_args.append(s//8)
                llvm_args_types.append(ctypes.c_longlong)
            if i >= num_in:
                all_outputs.append(ndarray)
        # Ndarray is sparse
        else:
            # Working on the output matrix
            if i >= num_in:
                A1pos = memref_i64()
                A1crd = memref_i64()
                A1tile_pos = memref_i64()
                A1tile_crd = memref_i64()
                A2pos = memref_i64()
                A2crd = memref_i64()
                A2tile_pos = memref_i64()
                A2tile_crd = memref_i64()
                Aval = memref_f64()

                # llvm_args += [*expand_memref_ptr(A1pos), *expand_memref_ptr(A1crd), *expand_memref_ptr(A2pos), *expand_memref_ptr(A2crd), *expand_memref_ptr(Aval)]
                # llvm_args_types += [POINTER(memref_i64), POINTER(memref_i64), c_longlong, c_longlong, c_longlong] * 4 + [POINTER(memref_f64), POINTER(memref_f64), c_longlong, c_longlong, c_longlong]
                # all_outputs.append((A1pos, A1crd, A2pos, A2crd, Aval))
                
                # With tiles
                llvm_args += [*expand_memref_ptr(A1pos), *expand_memref_ptr(A1crd), *expand_memref_ptr(A1tile_pos), *expand_memref_ptr(A1tile_crd), *expand_memref_ptr(A2pos), *expand_memref_ptr(A2crd), *expand_memref_ptr(A2tile_pos), *expand_memref_ptr(A2tile_crd), *expand_memref_ptr(Aval)]
                llvm_args_types += [POINTER(memref_i64), POINTER(memref_i64), c_longlong, c_longlong, c_longlong] * 8 + [POINTER(memref_f64), POINTER(memref_f64), c_longlong, c_longlong, c_longlong]
                all_outputs.append((A1pos, A1crd, A1tile_pos, A1tile_crd, A2pos, A2crd, A2tile_pos, A2tile_crd, Aval))
            else:
                # [TODO] The arrays used as inputs for the comet generated code need to be updated to take into account the extra tile component
                A1tile_pos = np.array([-1], dtype=np.int64)
                A1tile_crd = np.array([-1], dtype=np.int64)
                A2tile_pos = np.array([-1], dtype=np.int64)
                A2tile_crd = np.array([-1], dtype=np.int64)
                # CSR
                if ndarray.format == 'csr':
                    A1pos = np.array([ndarray.shape[0]], dtype=np.int64)
                    A1crd = np.array([-1], dtype=np.int64)
                    A2pos = ndarray.indptr.astype('int64')
                    A2crd = ndarray.indices.astype('int64')

                    # Based on the desc_sizes array in SparseUtils.cpp:read_input_sizes_2D
                
                    # llvm_args += [*np_array_to_memref(np.array([1, 1, ndarray.shape[0] + 1, ndarray.nnz, ndarray.nnz, ndarray.shape[0], ndarray.shape[1]], dtype='int64'))]
                    # With tiles
                    llvm_args += [*np_array_to_memref(np.array([1, 1, 0, 0, ndarray.shape[0] + 1, ndarray.nnz, 0, 0, ndarray.nnz, ndarray.shape[0], ndarray.shape[1]], dtype='int64'))]
                # COO
                elif ndarray.format == 'coo':
                    A1pos = np.array([0, ndarray.nnz], dtype=np.int64)
                    A1crd = ndarray.row.astype('int64')
                    A2pos = np.array([-1], dtype=np.int64)
                    A2crd = ndarray.col.astype('int64')

                    # Based on the desc_sizes array in SparseUtils.cpp:read_input_sizes_2D
                
                    # llvm_args += [*np_array_to_memref(np.array([2, ndarray.nnz, 1, ndarray.nnz, ndarray.nnz, ndarray.shape[0], ndarray.shape[1]], dtype='int64'))]
                    # With tiles
                    llvm_args += [*np_array_to_memref(np.array([2, ndarray.nnz, 0, 0, 1, ndarray.nnz, 0, 0, ndarray.nnz, ndarray.shape[0], ndarray.shape[1]], dtype='int64'))]
                
                # CSC
                elif ndarray.format == 'csc':
                    A1pos = ndarray.indptr.astype('int64')
                    A1crd = ndarray.indices.astype('int64')
                    A2pos = np.array([ndarray.shape[1]], dtype=np.int64)
                    
                    # Based on the desc_sizes array in SparseUtils.cpp:read_input_sizes_2D
                
                    # llvm_args += [*np_array_to_memref(np.array([ndarray.shape[1] + 1, ndarray.nnz, 1, 1, ndarray.nnz, ndarray.shape[0], ndarray.shape[1]], dtype='int64'))]
                    # With tiles
                    llvm_args += [*np_array_to_memref(np.array([ndarray.shape[1] + 1, ndarray.nnz, 0, 0, 1, 1, 0, 0, ndarray.nnz, ndarray.shape[0], ndarray.shape[1]], dtype='int64'))]
                
                Aval = ndarray.data.astype('float64')
                # Based on the  desc_A1pos/crd, desc_A2pos/crd, desc_Aval arrays in SparseUtils.cpp: read_input_2D
                # Expand to memrefs llvmir implementation
                
                # llvm_args += [*np_array_to_memref(A1pos),  *np_array_to_memref(A1crd), *np_array_to_memref(A2pos), *np_array_to_memref(A2crd), *np_array_to_memref(Aval)]
                # With tiles
                llvm_args += [*np_array_to_memref(A1pos),  *np_array_to_memref(A1crd), *np_array_to_memref(A1tile_pos), *np_array_to_memref(A1tile_crd), *np_array_to_memref(A2pos), *np_array_to_memref(A2crd), *np_array_to_memref(A2tile_pos), *np_array_to_memref(A2tile_crd), *np_array_to_memref(Aval)]
                
                # Set the datatypes expected from the function in the shared library.
                # If we don't define this the data are not passed correctly
                
                # llvm_args_types += [ctypes.POINTER(c_longlong), ctypes.POINTER(c_longlong), c_longlong, c_longlong, c_longlong] * 5  + [ctypes.POINTER(c_double), ctypes.POINTER(c_double), c_longlong, c_longlong, c_longlong]
                # With tiles
                llvm_args_types += [ctypes.POINTER(c_longlong), ctypes.POINTER(c_longlong), c_longlong, c_longlong, c_longlong] * 9  + [ctypes.POINTER(c_double), ctypes.POINTER(c_double), c_longlong, c_longlong, c_longlong]

    return llvm_args, llvm_args_types, all_outputs

#Translating llvm dialect to llvm IR using mlir-translate and then executing the IR using lli
def translate_and_exec_llvm_with_jit(llvm_in,scf_lower_flags, func_name, inputs, outputs, uuid_s):


    llvmir_file = uuid_s+'.ll'

    # path_to_cometopt = cfg.comet_path+"/bin/comet-opt"
    path_to_cometopt = cfg.comet_path+"/bin/comet-opt -x mlir"
    to_llvm_command = path_to_cometopt + scf_lower_flags #+ llvm_in
    translate_mlir_command = cfg.llvm_path+"/bin/mlir-translate --mlir-to-llvmir -- " 
    libname =   "./lib"+llvmir_file+func_name+".so"
    gcc_command = cfg.llvm_path+"/bin/clang -march=native -mtune=native -x ir -Wno-everything --shared -O3 "+platform_args+ " -o "+ temp_dir + libname+" -fpic -L {0}/lib/ -Wl,-rpath,{0}/lib/ -lcomet_runner_utils -".format(cfg.comet_path)

    # We merge all several calls in a single call to the shell in order to only pay the overhead of process creation once.
    # 1. Call comet to lower scf code to llvm
    # 2. Call mlir-translate to convert llvm to llvmir 
    # 3. Call clang to generate library
    p = subprocess.run(to_llvm_command +' 2>&1 |  '+ translate_mlir_command +' | ' + gcc_command , input=llvm_in.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if(p.returncode != 0):
        cleanup()
        raise AssertionError("gcc failed with error code: {}. Error: {}".format(p.returncode, p.stderr))
    files_to_cleanup.append(os.path.join( os.getcwd(), temp_dir +libname))

    # Load code generated from COMET
    lib = ctypes.cdll.LoadLibrary(temp_dir +libname)
    func = lib.__getattr__(func_name)

    # Get the inputs, input types and the output containers
    args, arg_types, all_output = generate_llvm_args_from_ndarrays(len(inputs),*(inputs), *(outputs))
    func.argtypes = arg_types

    # Uncomment to measure execution time without the compilation process
    
    # start = time.time()
    func(*(args))
    # end = time.time()
    # print("Kernel execution time JIT: {}".format(end-start))


    out = None
    ret_outputs = []
    for v0, v1 in zip(all_output, outputs):
        if scp.sparse.issparse(v1):
            # A1pos, A1crd, A2pos, A2crd, Aval = v0
            # With tiles
            A1pos, A1crd, A1tile_pos, A1tile_crd, A2pos, A2crd, A2tile_pos, A2tile_crd, Aval = v0
            if v1.format == 'csr':
                np_r_indices = np.ctypeslib.as_array(A2pos.mem, [A2pos.dim])
                np_c_indices = np.ctypeslib.as_array(A2crd.mem, [A2crd.dim])
                np_values = np.ctypeslib.as_array(Aval.mem, [Aval.dim])
                ret_outputs.append(scp.sparse.csr_array((np_values, np_c_indices, np_r_indices), copy=False))
            elif v1.format == 'coo':
                rows = np.ctypeslib.as_array(A1crd.mem, [A1crd.dim])
                cols = np.ctypeslib.as_array(A2crd.mem, [A2crd.dim])
                np_values = np.ctypeslib.as_array(Aval.mem, [Aval.dim])
                ret_outputs.append(scp.sparse.coo_array((np_values, (rows, cols)), copy=False))
        else:
            if v1.shape == (1,):
                ret_outputs.append(np.squeeze(v1))
            else:
                ret_outputs.append(v1)

    if len(ret_outputs) == 1:
        out =  ret_outputs.pop()
    else:
        out = ret_outputs

    return out, llvmir_file

def func_execute(func, args):
    func(*(args))

def translate_and_exec_llvm(llvm_in,func_name, out_dims, uuid_s):

    translate_mlir_command = "../llvm/build/bin/mlir-translate --mlir-to-llvmir " + llvm_in

    p = subprocess.run(shlex.split(translate_mlir_command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)
    llvmir_out = p.stdout.decode()

    llvmir_out = llvmir_out.replace(func_name, "main")

    llvmir_file = uuid_s+'.ll'

    if(os.path.exists(llvmir_file) == False):
        f = open(os.path.join( os.getcwd(), llvmir_file), 'w')
        files_to_cleanup.append(os.path.join( os.getcwd(), llvmir_file))
    else:
        f = open(llvmir_file, 'w')

    # with open(os.path.join( os.getcwd(),llvmir_file), 'wb') as f:
    f.write(llvmir_out)
    f.close()

    # lli_command = "../llvm/build/bin/lli -load " + comet_runner_util  + " "+ llvmir_file
    lli_command = "../llvm/build/bin/mlir-cpu-runner " + llvm_in + " -O3 -e "+func_name+" -entry-point-result=void -shared-libs=../build/lib/libcomet_runner_utils.dylib"
    # print(lli_command)
    start = time.time()
    p = subprocess.run(shlex.split(lli_command) , stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=False,close_fds=False)
    result = p.stdout
    result_str = result.decode("ascii").replace("\n","").strip().split("data =")

    output_arrays_list = []
    indx = 0
    for str_out in result_str:
        str_out = str_out.strip()
        output_array = np.fromstring(str_out, dtype=float, sep=',')
        
        if(output_array.size > 0):
            output_array = output_array.reshape(tuple(out_dims[indx]))
            output_arrays_list.append(output_array)
            indx = indx + 1
    end = time.time()
    print("Kernel execution time no JIT: {}".format(end-start))

    if(len(output_arrays_list) > 1):
        return tuple(output_arrays_list),llvmir_file
    else:
        return output_arrays_list.pop(),llvmir_file

def lower_dialect(ta_dialect_rep, out_dims, compile_with_flags,func_name):

    #lower TA dialect to the SCF dialect
    mlir_lower_flags = ""

    if isinstance(compile_with_flags,tuple):
        for i in range(len(compile_with_flags)):
            mlir_lower_flags += compile_with_flags[i] + " "
    
    elif(isinstance(compile_with_flags,str)):
        mlir_lower_flags += compile_with_flags 
    
    mlir_lower_flags += " --convert-ta-to-it --convert-to-loops "

    # scf_lower_flags =  " --lower-affine --convert-linalg-to-loops --convert-scf-to-std --convert-linalg-to-llvm --convert-std-to-llvm "
    scf_lower_flags =  " --convert-to-llvm "

    if("-emit-ta" in mlir_lower_flags):
        print(ta_dialect_rep)
        return

     #write the TA dialect rep to file
    uuid_s = str(uuid.uuid4())
    ta_dialect_file = uuid_s+'.mlir'
    # print("uuid_s: ", uuid_s)
    if(os.path.exists(ta_dialect_file) is False):
        f = open(os.path.join( os.getcwd(), ta_dialect_file), 'w')
        files_to_cleanup.append(os.path.join( os.getcwd(), ta_dialect_file))
    else:
        f = open(ta_dialect_file, 'w')
    
    f.write(ta_dialect_rep)
    f.close()

    # Uncomment for debugging pusposes
    scf_out_file = lower_ta_to_mlir(ta_dialect_file, mlir_lower_flags, uuid_s)
    # scf_out_file = lower_ta_to_mlir_with_jit(ta_dialect_file, mlir_lower_flags, args_vals)
    
    # Running --convert-ta-to-it --convert-to-loops and  --convert-to-llvm in separate steps 
    # does not produce correct output. This is an issue with the backend.

    #lower the SCF dialect to first STD dialect and then to the llvm dialect
    llvm_out_file = lower_scf_to_llvm(scf_out_file, scf_lower_flags, uuid_s)
    # llvm_out_file = lower_scf_to_llvm(ta_dialect_file, mlir_lower_flags + scf_lower_flags)
   
    result,llvmir_file = translate_and_exec_llvm(llvm_out_file,func_name, out_dims, uuid_s)
    
    return result


def lower_dialect_with_jit(ta_dialect_rep, target: str, out_dims, compile_with_flags,func_name, args_vals, outputs):

    mlir_lower_flags = " "
        
    if compile_with_flags != None:
        if "--convert-tc-to-ttgt" not in compile_with_flags:
            mlir_lower_flags += "  --convert-ta-to-it "
        if "--opt-fusion"  in compile_with_flags:
            mlir_lower_flags += "--opt-fusion"
            compile_with_flags = compile_with_flags.replace("--opt-fusion","")
            compile_with_flags = compile_with_flags.replace("--opt-comp-workspace","")
        if "-opt-matmul-tiling" not in compile_with_flags:
            mlir_lower_flags += "   --convert-to-loops "
        mlir_lower_flags =" "+compile_with_flags + mlir_lower_flags
    else:
        mlir_lower_flags = "  --convert-ta-to-it --convert-to-loops "
    # scf_lower_flags =  " --lower-affine --convert-linalg-to-loops --convert-scf-to-std --convert-linalg-to-llvm --convert-std-to-llvm "
    scf_lower_flags =  " --convert-to-llvm "
    
    if target != "cpu":
        if target.startswith("sm_") or target.startswith("compute_") or target.startswith("lto_"):
            if not cfg.gpu_target_enabled:
                raise "COMET gpu target is not enabled"
            
            scf_lower_flags += " " + " --convert-to-triton --target=GPU --gpu-compute-capability="+target.split("_")[1]
            mlir_lower_flags += " " + "--target=GPU"
        elif target == "gpu":
            if not cfg.gpu_target_enabled:
                raise "COMET gpu target is not enabled"
        
            scf_lower_flags += " " + " --convert-to-triton --target=GPU"
            mlir_lower_flags += " " + "--target=GPU"
        else :
            raise "Expected target formats:\
                    cpu, compute_<version>, sm_<version>, lto_<version>"
    
    if("-emit-ta" in mlir_lower_flags):
        print(ta_dialect_rep)
        return

    uuid_s = str(uuid.uuid4())
    ta_dialect_file = temp_dir+uuid_s+'.mlir'
    # if(os.path.exists(ta_dialect_file) == False):
    #     f = open(os.path.join( os.getcwd(), ta_dialect_file), 'w')
    #     files_to_cleanup.append(os.path.join( os.getcwd(), ta_dialect_file))
    # else:
    #     f = open(ta_dialect_file, 'w')
    
    # f.write(ta_dialect_rep)
    # f.close()

    # Convert TA to SCF
    # scf_out_file = lower_ta_to_mlir_with_jit(ta_dialect_file, mlir_lower_flags, args_vals, uuid_s)
    scf_out_file = lower_ta_to_mlir_with_jit(ta_dialect_rep, mlir_lower_flags, args_vals, uuid_s)

    #lower the SCF dialect to LLVMIR and execute
    result,llvmir_file = translate_and_exec_llvm_with_jit(scf_out_file, scf_lower_flags, func_name, args_vals, outputs, uuid_s)

    return result