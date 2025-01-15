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

def create_memref_type(type, rank):
    class memref_template(Structure):
        _fields_ = [ ('mem_aligned', POINTER(type)), ('mem', POINTER(type)), ('offset', c_int64), ('dims', c_int64 * rank), ('stride', c_int64 * rank)]
    
    return memref_template

class memref_i64(Structure):
    _fields_ = [ ('mem_aligned', POINTER(c_int64)), ('mem', POINTER(c_int64)), ('offset', c_int64), ('dim', c_int64 * 1), ('stride', c_int64 * 1)]

class memref_i32(Structure):
    _fields_ = [ ('mem_aligned', POINTER(c_int32)), ('mem', POINTER(c_int32)), ('offset', c_int64), ('dim', c_int64 * 1), ('stride', c_int64 * 1)]

class memref_f64(Structure):
    _fields_ = [ ('mem_aligned', POINTER(c_double)), ('mem', POINTER(c_double)), ('offset', c_int64), ('dim', c_int64 * 1), ('stride', c_int64 * 1)]

class memref_f32(Structure):
    _fields_ = [ ('mem_aligned', POINTER(c_float)), ('mem', POINTER(c_float)), ('offset', c_int64), ('dim', c_int64 * 1), ('stride', c_int64 * 1)]

def memref_from_np_array(np_array):
    ctype = None
    if np_array.dtype == 'int64':
        ctype = c_int64
        if(len(np_array.shape) == 1):
            constructor = memref_i64
        else: 
            constructor = create_memref_type(ctype,len(np_array.shape))
    elif np_array.dtype == 'int32':
        ctype = c_int32
        if(len(np_array.shape) == 1):
            constructor = memref_i32
        else: 
            constructor = create_memref_type(ctype,len(np_array.shape))
    elif np_array.dtype == 'float32':
        ctype = c_float
        if(len(np_array.shape) == 1):
            constructor = memref_f32
        else: 
            constructor = create_memref_type(ctype,len(np_array.shape))
    elif np_array.dtype == 'float64':
        ctype = c_double
        if(len(np_array.shape) == 1):
            constructor = memref_f64
        else: 
            constructor = create_memref_type(ctype,len(np_array.shape))
    return constructor(np_array.ctypes.data_as(ctypes.POINTER(ctype)), np_array.ctypes.data_as(ctypes.POINTER(ctype)), 0, (c_int64*len(np_array.shape))(*np_array.shape), (c_int64*len(np_array.shape))(*[s//8 for s in np_array.strides])), constructor

# llvm_args += [*expand_memref_ptr(dim_sizes), 0, *expand_memref_ptr(A1pos), 0, *expand_memref_ptr(A2pos), *expand_memref_ptr(A2crd), *expand_memref_ptr(Aval)]

class output_csr_f64_i64(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i64), ('insert_2', c_int64), ('A2pos', memref_i64), ('A2crd', memref_i64), ('Aval', memref_f64)]
    
class output_csr_f32_i64(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i64), ('insert_2', c_int64), ('A2pos', memref_i64), ('A2crd', memref_i64), ('Aval', memref_f32)]

class output_coo_f64_i64(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i64), ('A1crd', memref_i64), ('insert_2', c_int64), ('A2crd', memref_i64), ('Aval', memref_f64)]

class output_coo_f32_i64(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i64), ('A1crd', memref_i64), ('insert_2', c_int64), ('A2crd', memref_i64), ('Aval', memref_f32)]

class output_csr_f64_i32(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i32), ('insert_2', c_int64), ('A2pos', memref_i32), ('A2crd', memref_i32), ('Aval', memref_f64)]
    
class output_csr_f32_i32(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i32), ('insert_2', c_int64), ('A2pos', memref_i32), ('A2crd', memref_i32), ('Aval', memref_f32)]

class output_coo_f64_i32(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i32), ('A1crd', memref_i32), ('insert_2', c_int64), ('A2crd', memref_i32), ('Aval', memref_f64)]

class output_coo_f32_i32(Structure):
    _fields_ = [('dims_sizes', memref_i64), ('insert_1', c_int64), ('A1pos', memref_i32), ('A1crd', memref_i32), ('insert_2', c_int64), ('A2crd', memref_i32), ('Aval', memref_f32)]

def np_array_to_memref(np_array):
    ctype = ctypes.c_int64
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

    # scf_out = comment_unneeded_sparse(scf_out, arg_vals)
    # scf_out = comment_unneeded_dense(scf_out, arg_vals)
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
def memref_index_type():
    return [ctypes.POINTER(c_int64), ctypes.POINTER(c_int64), c_int64, c_int64, c_int64]

def memref_f64_type():
    return [ctypes.POINTER(c_double), ctypes.POINTER(c_double), c_int64, c_int64, c_int64]

def generate_llvm_args_from_ndarrays(num_in, *ndargs):
    llvm_args = []
    llvm_args_types = []
    all_outputs = []
    aux = [] # Used to make sure data created in this function are not freed prematurely.
    for i, ndarray in enumerate(ndargs):
        # Ndarray is dense
        if not scp.sparse.issparse(ndarray):
            memref, type = memref_from_np_array(ndarray)
            llvm_args.append(memref)
            llvm_args_types.append(POINTER(type))

            if i >= num_in:
                all_outputs.append(ndarray)
        # Ndarray is sparse
        else:
            dims = np.array(ndarray.shape, dtype=np.int64)
            dim_sizes, dim_type = memref_from_np_array(dims)
            aux.append(dims)
            Aval, Aval_type = memref_from_np_array(ndarray.data)

            # Working on the output matrix
            if i >= num_in:
                if ndarray.format == 'csr':
                    A1pos_temp = np.array([ndarray.shape[0]], dtype=ndarray.indptr.dtype)
                    aux.append(A1pos_temp) # Make sure the array we created persists until we actually call the mlir-generated function
                    A1pos, A1pos_type = memref_from_np_array(A1pos_temp)
                    A2pos, A2pos_type = memref_from_np_array(ndarray.indptr)
                    A2crd, A2crd_type = memref_from_np_array(ndarray.indices)


                    if ndarray.data.dtype == 'float64':
                        if ndarray.indices.dtype == 'int32':
                            out_type = output_csr_f64_i32
                        else:
                            out_type = output_csr_f64_i64
                    elif ndarray.data.dtype == 'float32':
                        if ndarray.indices.dtype == 'int32':
                            out_type = output_csr_f32_i32
                        else:
                            out_type = output_csr_f32_i64

                    out = out_type(dim_sizes, 0, A1pos, 0, A2pos, A2crd, Aval)
                elif ndarray.format == 'coo':
                    A1pos_temp = np.array([0, ndarray.nnz], dtype=ndarray.row.dtype)
                    aux.append(A1pos_temp) # Make sure the array we created persists until we actually call the mlir-generated function
                    A1pos, A1pos_type = memref_from_np_array(A1pos_temp)
                    A1crd, A1crd_type = memref_from_np_array(ndarray.row)
                    A2crd, A2crd_type = memref_from_np_array(ndarray.col)

                    if ndarray.data.dtype == 'float64':
                        if ndarray.row.dtype == 'int32':
                            out_type = output_coo_f64_i32
                        else:
                            out_type = output_coo_f64_i64
                    elif ndarray.data.dtype == 'float32':
                        if ndarray.row.dtype == 'int32':
                            out_type = output_coo_f32_i32
                        else:
                            out_type = output_coo_f32_i64

                    out = out_type(dim_sizes, 0, A1pos, A1crd, 0, A2crd, Aval)
                else :
                    raise Exception("Unsupported sparse matrix type")

                llvm_args_types = [POINTER(out_type)] + llvm_args_types
                llvm_args = [out] + llvm_args
                all_outputs.append(out)
            else:

                
                insert_pos_1 = 0
                insert_pos_2 = 0
                # CSR
                if ndarray.format == 'csr':
                    A1tile_pos = np.array([-1], dtype=ndarray.indptr.dtype)
                    A1tile_crd = np.array([-1], dtype=ndarray.indptr.dtype)
                    A2tile_pos = np.array([-1], dtype=ndarray.indptr.dtype)
                    A2tile_crd = np.array([-1], dtype=ndarray.indptr.dtype)
                    A1pos_temp = np.array([ndarray.shape[0]], dtype=ndarray.indptr.dtype)
                    aux.append(A1pos_temp) # Make sure the array we created persists until we actually call the mlir-generated function
                    A1pos, A1pos_type = memref_from_np_array(A1pos_temp)
                    A2pos, A2pos_type = memref_from_np_array(ndarray.indptr)
                    A2crd, A2crd_type = memref_from_np_array(ndarray.indices)

                    # Based on the desc_sizes array in SparseUtils.cpp:read_input_sizes_2D
                    llvm_args += [dim_sizes, insert_pos_1, A1pos, insert_pos_2, A2pos, A2crd]
                    llvm_args_types += [POINTER(dim_type)] + [c_int64] + [POINTER(A1pos_type)] + [c_int64] + [POINTER(A2pos_type), POINTER(A2crd_type)]
                # COO
                elif ndarray.format == 'coo':
                    A1tile_pos = np.array([-1], dtype=ndarray.row.dtype)
                    A1tile_crd = np.array([-1], dtype=ndarray.row.dtype)
                    A2tile_pos = np.array([-1], dtype=ndarray.row.dtype)
                    A2tile_crd = np.array([-1], dtype=ndarray.row.dtype)
                    A1pos_temp = np.array([0, ndarray.nnz], dtype=ndarray.row.dtype)
                    aux.append(A1pos_temp) # Make sure the array we created persists until we actually call the mlir-generated function
                    A1pos, A1pos_type = memref_from_np_array(A1pos_temp)
                    A1crd, A1crd_type = memref_from_np_array(ndarray.row)
                    A2crd, A2crd_type = memref_from_np_array(ndarray.col)

                    llvm_args += [dim_sizes, insert_pos_1, A1pos,  A1crd, insert_pos_2, A2crd]
                    llvm_args_types += [POINTER(dim_type)] + [c_int64] + [POINTER(A1pos_type), POINTER(A1crd_type)] + [c_int64] + [POINTER(A2crd_type)]
                # # CSC
                # elif ndarray.format == 'csc':
                #     A1pos = ndarray.indptr.astype('int64')
                #     A1crd = ndarray.indices.astype('int64')
                #     A2pos = np.array([ndarray.shape[1]], dtype=np.int64)
                    
                #     # Based on the desc_sizes array in SparseUtils.cpp:read_input_sizes_2D
                
                #     # llvm_args += [*np_array_to_memref(np.array([ndarray.shape[1] + 1, ndarray.nnz, 1, 1, ndarray.nnz, ndarray.shape[0], ndarray.shape[1]], dtype='int64'))]
                #     # With tiles
                #     llvm_args += [*np_array_to_memref(np.array([ndarray.shape[1] + 1, ndarray.nnz, 0, 0, 1, 1, 0, 0, ndarray.nnz, ndarray.shape[0], ndarray.shape[1]], dtype='int64'))]
                
                # Based on the  desc_A1pos/crd, desc_A2pos/crd, desc_Aval arrays in SparseUtils.cpp: read_input_2D
                # Expand to memrefs llvmir implementation
                
                llvm_args += [Aval]
                llvm_args_types += [POINTER(Aval_type)]

    return llvm_args, llvm_args_types, all_outputs, aux

#Translating llvm dialect to llvm IR using mlir-translate and then executing the IR using lli
def translate_and_exec_llvm_with_jit(llvm_in,scf_lower_flags, func_name, inputs, outputs, uuid_s):


    llvmir_file = uuid_s+'.ll'
    llvm_in = llvm_in.replace('call @comet_print_memref_', '//call @comet_print_memref_')
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
    # p = subprocess.run(to_llvm_command, input=llvm_in.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    p = subprocess.run(to_llvm_command +' 2>&1 |  '+ translate_mlir_command +' | ' + gcc_command , input=llvm_in.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if(p.returncode != 0):
        cleanup()
        raise AssertionError("gcc failed with error code: {}. Error: {}".format(p.returncode, p.stdout))
    files_to_cleanup.append(os.path.join( os.getcwd(), temp_dir +libname))

    # Load code generated from COMET
    lib = ctypes.cdll.LoadLibrary(temp_dir +libname)
    func = lib.__getattr__("_mlir_ciface_"+func_name)

    # Get the inputs, input types and the output containers
    args, arg_types, all_output, aux = generate_llvm_args_from_ndarrays(len(inputs),*(inputs), *(outputs))
    func.argtypes = arg_types
    if outputs[0].shape == (1,):
        func.restype = c_double
    # Uncomment to measure execution time without the compilation process
    # start = time.time()
    ret = func(*[byref(arg) if not isinstance(arg, int) else arg for arg in args])
    # end = time.time()
    # print("Kernel execution time JIT: {}".format(end-start))

    out = None
    ret_outputs = []
    for v0, v1 in zip(all_output, outputs):
        if scp.sparse.issparse(v1):
            if v1.format == 'csr':
                out_csr = v0
                np_r_indices = np.ctypeslib.as_array(out_csr.A2pos.mem, [out_csr.A2pos.dim[0]])
                np_c_indices = np.ctypeslib.as_array(out_csr.A2crd.mem, [out_csr.A2crd.dim[0]])
                np_values = np.ctypeslib.as_array(out_csr.Aval.mem, [out_csr.Aval.dim[0]])
                ret_outputs.append(scp.sparse.csr_array((np_values, np_c_indices, np_r_indices), copy=False))
            elif v1.format == 'coo':
                out_coo = v0
                rows = np.ctypeslib.as_array(out_coo.A1crd.mem, [out_coo.A1crd.dim[0]])
                cols = np.ctypeslib.as_array(out_coo.A2crd.mem, [out_coo.A2crd.dim[0]])
                np_values = np.ctypeslib.as_array(out_coo.Aval.mem, [out_coo.Aval.dim[0]])
                ret_outputs.append(scp.sparse.coo_array((np_values, (rows, cols)), copy=False))
        else:
            if v1.shape == (1,):
                ret_outputs.append(ret)
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
                raise Exception("COMET gpu target is not enabled")
            
            scf_lower_flags += " " + " --convert-to-triton --target=GPU --gpu-compute-capability="+target.split("_")[1]
            mlir_lower_flags += " " + "--target=GPU"
        elif target == "gpu":
            if not cfg.gpu_target_enabled:
                raise Exception("COMET gpu target is not enabled")
        
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