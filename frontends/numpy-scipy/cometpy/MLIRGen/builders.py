import jinja2
import functools
import itertools

from typing import Dict, List, Tuple, Sequence, Union
from collections import OrderedDict
from ast import operator
from cometpy.MLIRGen import types_mlir
from cometpy.MLIRGen.types import *

def get_tensor_type(datatype, shape, format, indices_type):
    if format != DENSE:
        tensor_formats = []
        if format == CSR:
            tensor_formats.append("d")
            tensor_formats.append("unk")
            tensor_formats.append("cu")
            tensor_formats.append("unk")
        elif format == COO:
            tensor_formats.append("cn")
            tensor_formats.append("unk")
            tensor_formats.append("s")
            tensor_formats.append("unk")
        return "!ta.sparse_tensor<{}, {}, {}, {}>".format(datatype, indices_type,"x".join(str(v) for v in shape), ",".join(f for f in tensor_formats))
    else:
        return "tensor<{}x{}>".format("x".join(str(v) for v in shape), datatype)

class Dialect:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{self.name} dialect"


class MLIRFunctionBuilder:
    _ops = {}

    default_indentation_size = 4
    indentation_delta_size = 2
    module_wrapper_text = jinja2.Template(
        "{{ aliases }}module {\n   {{ body }}\nfunc.func private @quick_sort(memref<*xindex>, index)\n}\n",
        undefined=jinja2.StrictUndefined,
    )
    function_wrapper_text = jinja2.Template(
        ("" * default_indentation_size)
        + "func.func {% if private_func %}private {% endif %}@{{func_name}}({{signature}})  -> {{return_type}} {% if return_type %} attributes {llvm.emit_c_interface} {% endif %}{"
        + "\n"
        + "{{statements}}"
        + "\n"
        + (" " * default_indentation_size)
        + "}",
        undefined=jinja2.StrictUndefined,
    )

    def __init__(
        self,
        func_name: str,
        input_types,
        return_types, #Sequence[Union[str, types_mlir.Type]],
        aliases: types_mlir.AliasMap = None,
    ) -> None:
        # TODO mlir functions can return zero or more results https://mlir.llvm.org/docs/LangRef/#operations
        # handle the cases where the number of return types is not 1
        if aliases is None:

            aliases = types_mlir.AliasMap()
        self.aliases = aliases

        # Build input vars and ensure all types are proper Types
        # inputs = []
        # for i, it in enumerate(input_types):
        #     it = Type.find(it, aliases)
        #     # Create initialized MLIRVar
        #     iv = MLIRVar(f"arg{i}", it)
        #     iv._initialized = True
        #     inputs.append(iv)
        # return_types = [types_mlir.Type.find(rt, aliases) for rt in return_types]

        self.func_name = func_name
        self.inputs = input_types
        self.return_types = return_types

        self.var_name_counter = itertools.count()
        self.function_body_statements: List[str] = []
        self.temporary_statement_lists: List[List[str]] = []

        # function_name -> (function_mlir_definition, input_mlir_types, return_mlir_type)
        self.needed_function_table: Dict[
            str, Tuple[str, List[str], str]
        ] = OrderedDict()

        self.indentation_level = 1
        self._initialize_ops()

    def _initialize_ops(self):
        for dialect, ops in self._ops.items():
            if dialect is None:
                attach_point = self
            else:
                attach_point = getattr(self, dialect, None)
                if attach_point is None:
                    attach_point = Dialect(dialect)
                    setattr(self, dialect, attach_point)

            for opclass in ops.values():

                def op(opclass, *args, **kwargs):
                    ret_val, mlir = opclass.call(self, *args, **kwargs)
                    self.add_statement(mlir)
                    return ret_val

                func = functools.partial(op, opclass)
                setattr(attach_point, opclass.name, func)
    
    #######################################
    # MLIR Generation/Compilation Methods #
    #######################################

    def get_mlir_module(self, make_private=False):
        """Get the MLIR text for this function wrapped in a MLIR module with
        declarations of external helper functions."""
        aliases = "\n".join(
            f"#{name} = {typ.to_pretty_string()}" for name, typ in self.aliases.items()
        )
        body = self.get_mlir(make_private=make_private)
        return self.module_wrapper_text.render(aliases=aliases, body=body)

    def get_mlir(self, make_private=True, include_func_defs=True) -> str:
        if include_func_defs:
            needed_function_definitions = "\n    ".join(
                func_def for func_def, _, _ in self.needed_function_table.values()
            )
        else:
            needed_function_definitions = ""

        if len(self.temporary_statement_lists) > 0:
            raise RuntimeError(
                "Cannot get MLIR code while using temporary statement storage."
            )
        joined_statements = "\n".join(self.function_body_statements)

        return_type = ", ".join(str(rt) for rt in self.return_types)
        if len(self.return_types) != 1:
            return_type = f"({return_type})"
        signature = ", ".join(f"{ var[0]}: {var[1]}" if 'tensor' in var[1] else  f"{ var[0].replac }: tensor<1xf64>" if var[1] =="f64" else f"{ var[0]}: {var[1]}" for var in self.inputs)

        return needed_function_definitions + self.function_wrapper_text.render(
            private_func=make_private,
            func_name=self.func_name,
            signature=signature,
            return_type=return_type,
            statements=joined_statements,
        )


    def compile(self):
        # if engine is None:
        #     engine = self.engine

        # Force recompilation if name is already registered
        # if self.func_name in engine.name_to_callable:
        #     del engine.name_to_callable[self.func_name]

        mlir = self.get_mlir_module()

        # engine.add(mlir, passes)
        # func = engine[self.func_name]
        # func.builder = self

        return mlir

    
    ################################
    # MLIR Building Method Helpers #
    ################################

    @property
    def current_statement_list(self) -> List[str]:
        return (
            self.temporary_statement_lists[-1]
            if len(self.temporary_statement_lists) > 0
            else self.function_body_statements
        )
    
    def add_statement(self, statement: str) -> None:
        """In an ideal world, no human would ever call this method."""
        if not statement:
            return

        for line in map(str.strip, statement.split("\n")):
            self.current_statement_list.append(
                " " * self.default_indentation_size
                + " " * self.indentation_delta_size * self.indentation_level
                + line
            )
    

class TensorSumBuilder:
    indentation_size = 4
    tensor_sum_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + '%t{{lhs}} = "ta.reduce"{{operators}}'
        +' : ({{inputtype}})'
        +"-> {{output_types}}"
        + "\n" ,
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, data): # lhs, operators, tensors_shapes, label_map):
        self.lhs = data["out_id"]
        self.input_type = get_tensor_type(data['value_type'], data["shapes"][0],  data["formats"][0], data['indices_type'][0])
        self.output_type = data['value_type']

        self.operators = "({})".format(",".join("%t"+str(v) for v in data["operands"]))

    def build_op(self):
        output_type =  self.output_type
        
        return self.tensor_sum_wrapper_text.render(
            lhs = self.lhs,
            operators = self.operators,
            inputtype = self.input_type,
            output_types = output_type
        )

class SetOp_Builder:
    indentation_size = 4
    beta_val = 0.0

    set_op_wrapper_text = jinja2.Template(
        ("" * indentation_size)
        + '"ta.set_op"(%t{{input}},%t{{dest}}) {__beta__ = {{beta}} : f64} : ({{outputtype}}, {{outputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, data):# in_tensor, target, tensors_shapes, label_map, beta) :
        self.target = data["lhs"]
        self.in_tensor = data["rhs"]
        self.formats = data["formats"]
        self.tensors_shapes = data["shapes"]
        self.beta = "{:e}".format(data["beta"])
        self.type = data['value_type']
        self.indices_type = data['indices_type']


    def build_op(self):
        output_type = get_tensor_type(self.type, self.tensors_shapes[-1], self.formats[-1], self.indices_type[-1]) 

        return self.set_op_wrapper_text.render(
            dest = self.target,
            input = self.in_tensor,
            beta = self.beta,
            outputtype= output_type,

        )

class ScalarOp_Builder: 
    indentation_size = 4

    scalar_op_wrapper_text = jinja2.Template (
        ("" * indentation_size)
        +'%t{{dest}} = "ta.scalar"({{operators}})'
        +' <{op = "{{op}}"}> '
        +' : ({{inputtype}})'
        +' -> ({{outputtype}})'
        + "\n", 
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, data): 
        
        self.dest = data["out_id"]
        self.operators = "{}".format(",".join("%t"+str(v) for v in data["operands"]))
        self.tensors_shapes =[]
        for l in data["shapes"]:
            if isinstance(l, int):
                self.tensors_shapes.append(data['value_type'])
            else:
                self.tensors_shapes.append('tensor<1x{}>'.format(data['value_type']))

        self.op = data["op"]

    def build_op(self):
        input_type = []

        return self.scalar_op_wrapper_text.render(
            dest = self.dest,
            operators = self.operators,
            op = self.op,
            inputtype = ",".join(self.tensors_shapes[:2]),
            outputtype = self.tensors_shapes[-1],
        )    

class ArithOp_Builder:
    formats_str = ['Dense', 'CSR', 'COO', 'CSC']
    indentation_size = 4
    beta_val = 0.0



    tc_decl_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + '%t{{dest}} = "ta.mul"({{operators}})'
        + '{MaskType = "{{mask_type}}", ' 
        + '__alpha__ = 1.000000e+00 : f64, '  
        +"__beta__ = {{beta}} : f64,"
        + 'formats = [{{formats}}],'
        +'indexing_maps = {{indexing_maps}}, '
        +'operand_segment_sizes = array<i32:1, 1, {{lhs_dims}}, {{num_masks}}>, ' #[TODO] operand_segment_sizes should not be static
        +'semiring = "{{semiring}}"} : ' 
        +"({{inputtype}})"
        +"-> {{outputtype}}"
        + "\n" ,
        undefined=jinja2.StrictUndefined,
    )



    tensor_add_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        +'%t{{dest}} = "ta.add"({{operators}})'
        +' {'
        +' Masktype = "none",'
        +' formats = [{{formats}}],'
        +' indexing_maps = {{indexing_maps}},'
        +' semiring = "noop_plusxy"'
        +' }'
        +' : ({{inputtype}})'
        +"-> {{outputtype}}"
        + "\n", 
        # + '"ta.set_op"(%t{{dest}},%t{{dest}}): ({{outputtype}}, {{outputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    tensor_sub_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        +'%t{{dest}} = "ta.subtract"({{operators}})'
        +' {'
        +' Masktype = "none",'
        +' formats = [{{formats}}],'
        +' indexing_maps = {{indexing_maps}},'
        +' semiring = "noop_minus"'
        +' }'
        +' : ({{inputtype}})'
        +"-> {{outputtype}}"
        + "\n",
        # + '"ta.set_op"(%t{{dest}},%t{{dest}}): ({{outputtype}}, {{outputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    elewisemult_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        +'%t{{dest}} = "ta.elews_mul"({{operators}})'
        +' {__alpha__ = 1.000000e+00 : f64, '
        +"__beta__ = {{beta}}: f64,"
        + 'formats = [{{formats}}],'
        +'indexing_maps = {{indexing_maps}}, semiring = "{{semiring}}"} : '
        +"({{inputtype}})"
        +"-> {{outputtype}}"
        + "\n" ,
        # + '"ta.set_op"(%t{{dest}},%t{{dest}}): ({{outputtype}}, {{outputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    tranpose_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + '%t{{dest}} = "ta.transpose"({{operators}})'
        + '{__alpha__ = 1.000000e+00 : f64, '
        +"__beta__ = 0.000000e+00 : f64,"
        + 'formats = [{{formats}}],'
        +'indexing_maps = {{indexing_maps}},semiring = "plusxy_times"} : '
        +"({{inputtype}})"
        +"-> {{outputtype}}"
        + "\n" ,
        # + '"ta.set_op"(%temp_{{dest}},%t{{dest}}): ({{outputtype}}, {{outputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, data): 
        
        self.mask = None
        self.mask_type = "None"
        self.mask_shape = None
        self.semiring = None


        self.dest = data["out_id"]
        self.operators = "{}".format(",".join("%t"+str(v) for v in data["operands"])+","+",".join("%i"+str(vv) for v in data["op_ilabels"] for vv in v))
        self.tensors_shapes =[]
        self.op_ilabels = data["op_ilabels"]
        for l in data["shapes"]:
            self.tensors_shapes.append([ str(lbl) for lbl in l ]  )
        # self.tensors_shapes = [label_map[lbl][0] for lbl in tensors_shapes]
        self.opr_type = data["op_type"]
        # self.op = op
        self.formats = data["formats"]
        if "mask" in data:
            self.mask = data["mask"][0]
            self.mask_type = data["mask"][1]
            if data["mask"][2] != None:
                self.mask_shape = [ str(lbl) for lbl in data["mask"][2] ]
                self.operators+=",%t"+str(self.mask)
        if "semiring" in data:
            self.semiring = data["semiring"]
        self.beta = "{:e}".format(data["beta"])
        self.datatype = data["value_type"]
        self.indices_type = data['indices_type']

    def build_op(self):
        input_type = []
        for t,f,it in zip(self.tensors_shapes[:-1],self.formats[:-1], self.indices_type[:-1]):
            input_type.append(get_tensor_type(self.datatype, t, f, it))


        for t in self.tensors_shapes:
            for v in t:
                input_type.append("!ta.index")
        input_type = ",".join(input_type) 
        if self.mask_shape != None:

            input_type += ", " + get_tensor_type(self.datatype, self.mask_shape, CSR, self.indices_type[-1]) 
        # beta_val = ArithOp_Builder.get_beta_val(self.op)
        
        iMap = {}
        vMap = {}
        indexing_map = []
        i = 0
        temp = []
        for k, l in enumerate(self.op_ilabels[0]):
            iMap[l] = i
            vMap[l] = self.tensors_shapes[0][k]
            temp.append(i)
            i+=1
        
        indexing_map.append(temp)
        if len(self.op_ilabels) > 2:
            temp = []
            for k, l in enumerate(self.op_ilabels[1]):
                if l not in iMap:
                    iMap[l] = i
                    vMap[l] = self.tensors_shapes[1][k]
                    temp.append(i)
                    i+=1
                else:
                    temp.append(iMap[l])
            indexing_map.append(temp)
        temp = []

        for l in self.op_ilabels[-1]:
            temp.append(iMap[l])
        indexing_map.append(temp)
        indexing_maps = []

        output_type = get_tensor_type(self.datatype, [vMap[v] for v in self.op_ilabels[-1]], self.formats[-1], self.indices_type[-1]) 

        for imap in indexing_map:
            indexing_maps.append("affine_map<({})->({})>".format(",".join(["d"+str(l) for l in range(i)]) , ",".join(["d"+str(l) for l in imap])))
        
        indexing_maps = str(indexing_maps).replace("'","")

        # Tensor contraction
        if self.opr_type == 'c': 
            semiring = "plusxy_times"
            if self.semiring != None:
                s1, s2 = self.semiring.split(",")
                if s1 == "+":
                    semiring = "plusxy_"
                elif s1 == "any":
                    semiring = "any_"
                elif s1 == "min":
                    semiring = "minxy_"
                if s2 == "*":
                    semiring += "times"
                elif s2 == "pair":
                    semiring += "pairxy"
                elif s2 == "first":
                    semiring += "first"
                elif s2 == "+":
                    semiring += "plusxy"
                elif s2 == "second":
                    semiring += "second"


            return self.tc_decl_wrapper_text.render(
                    dest = self.dest,
                    operators = self.operators,
                    indexing_maps = indexing_maps,
                    inputtype = input_type,
                    outputtype = output_type,
                    # beta =  self.beta_val,
                    formats = '"{}", "{}", "{}"'.format(*[self.formats_str[x] for x in self.formats]),
                    lhs_dims = sum([len(t) for t in self.tensors_shapes ]),
                    semiring = semiring,
                    mask=self.mask,
                    mask_type = self.mask_type,
                    num_masks = 0 if self.mask == None else 1,
                    beta = self.beta,
                )
        # Add
        elif(self.opr_type == '+'):
            return self.tensor_add_wrapper_text.render(
                dest = self.dest,
                operators = self.operators,
                inputtype = input_type,
                outputtype = output_type,
                formats = '"{}", "{}", "{}"'.format(*[self.formats_str[x] for x in self.formats]),
                indexing_maps = indexing_maps
            )
        # Subtract
        elif(self.opr_type == '-'):
            return self.tensor_sub_wrapper_text.render(
                dest = self.dest,
                operators = self.operators,
                inputtype = input_type,
                outputtype = output_type,
                formats = '"{}", "{}", "{}"'.format(*[self.formats_str[x] for x in self.formats]),
                indexing_maps = indexing_maps
            )
        # Elementwise mult
        elif(self.opr_type == '*'):
            semiring = "noop_times"
            if self.semiring != None:
                if self.semiring == "min":
                    semiring = "noop_minxy"
                elif self.semiring == "-":
                    semiring = "noop_minus"
                elif self.semiring == "+":
                    semiring = "noop_plusxy"
                elif self.semiring == "*":
                    semiring = "noop_times"
            return self.elewisemult_wrapper_text.render(
                dest = self.dest,
                operators = self.operators,
                indexing_maps = indexing_maps,
                inputtype = input_type,
                outputtype = output_type,
                formats = '"{}", "{}", "{}"'.format(*[self.formats_str[x] for x in self.formats]),
                semiring = semiring,
                beta =  self.beta_val
            )
        # Transpose
        elif(self.opr_type == "t"):
            return self.tranpose_wrapper_text.render(
                dest = self.dest,
                operators = self.operators,
                indexing_maps = indexing_maps,
                inputtype = input_type,
                outputtype = output_type,
                # beta =  self.beta_val,
                formats = '"{}", "{}"'.format(*[self.formats_str[x] for x in self.formats]),
            )
    # def get_beta_val(op):
    #     if(op == '='):
    #         beta_val = '0.000000e+00'
    #     elif(op == '+='):
    #         beta_val = '1.000000e+00'
    #     elif(op == '-='):
    #         beta_val = '-1.000000e+00'
    #     elif(op == '+' or op == '-'):
    #         beta_val = '0.000000e+00'
    #     return beta_val
    

class Tensor_Decl_Builder:
    formats = ['Dense', 'CSR', 'COO', 'CSC']
    indentation_size = 4

    tensor_decl_wrapper_text = jinja2.Template(   
        ("" * indentation_size)
        + '%t{{lhs}} = "ta.{{decl}}ensor_decl"{{dims_tuple}}'
        + '{format = {{format}}} : '
        +"{{ranges_tuple}} -> "
        + "{{inputtype}}"
        + "\n" 
        +'"ta.fill{{where}}"(%t{{lhs}}) {{value}} : ({{inputtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    tensor_decl_wrapper_text_no_fill = jinja2.Template(   
        ("" * indentation_size)
        + '%t{{lhs}} = "ta.{{decl}}ensor_decl"{{dims_tuple}}'
        + '{format = {{format}}} : '
        +"{{ranges_tuple}} -> "
        + "{{inputtype}}"
        + "\n" ,
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, data)->None:
        self.lhs = data["id"]
        self.format = data["format"]
        self.inputtype = get_tensor_type(data["value_type"], data["shape"], self.format, data["indices_type"])

        # self.decl_vars = data["dimsSSA"]
        self.decl_vars = []
        self.is_input = data["is_input"]


    def build_tensor(self):
        dims_tuple = "({})".format(",".join("%d"+str(v) for v in self.decl_vars))
        ranges_tuple = "({})".format(",".join(["index"]* len(self.decl_vars)))

        if not self.format == DENSE:
            where = "_from_file"
            format = '"{}" , temporal_tensor = false'.format(self.formats[self.format])
            value = '{filename = "SPARSE_FILE_NAME0", readMode = 1 : i32}'
        else:
            where = ""
            format = '"{}"'.format(self.formats[self.format])
            value = '{value = 0.0 : f64}'

        if self.is_input or self.format == DENSE:
            return self.tensor_decl_wrapper_text.render(
                lhs = self.lhs,
                dims_tuple = dims_tuple,
                ranges_tuple = ranges_tuple,
                format = format,
                inputtype = self.inputtype,
                decl = "dense_t" if self.format == DENSE else "spT",
                where = where,
                value = value
            )
        else:
            return self.tensor_decl_wrapper_text_no_fill.render(
                lhs = self.lhs,
                dims_tuple = dims_tuple,
                ranges_tuple = ranges_tuple,
                format = format,
                inputtype = self.inputtype,
                decl = "dense_t" if self.format == DENSE else "spT",
            )

class PrintBuilder:
    indentation_size = 4

    tensor_print_text = jinja2.Template(
        ("" * indentation_size)
        +'"ta.print"(%t{{tensor}}) : ({{outtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, data): #operand, input_labels, dtype, label_map):
        self.operand = data["operands"][0]
        if data["shapes"] == 1 or data["shapes"] == [1]:
            self.outtype = data["value_type"]
        else:
            self.outtype = get_tensor_type(data['value_type'], data['shapes'][0], data['formats'][0], data['indices_type'][0]) 

    def build_op(self):
        return self.tensor_print_text.render(
            tensor = self.operand,
            outtype = self.outtype,
        )

class ReturnBuilder:
    indentation_size = 4

    tensor_print_text = jinja2.Template(
        ("" * indentation_size)
        +'"func.return"(%t{{tensor}}) : ({{outtype}}) -> ()\n',
        undefined=jinja2.StrictUndefined,
    )

    def __init__(self, data): #operand, input_labels, dtype, label_map):
        self.operand = data["operands"][0]
        if data["shapes"] == 1 or data["shapes"] == [1]:
            self.outtype = data["value_type"]
        else:
            self.outtype = get_tensor_type(data['value_type'], data['shapes'][0], data['formats'][0], data['indices_type'][0]) 

    def build_op(self):
        return self.tensor_print_text.render(
            tensor = self.operand,
            outtype = self.outtype,
        )