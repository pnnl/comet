// RUN: comet-opt %s --mem-alias-analysis 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "constants"
// CHECK-DAG: alloc_1#0 <-> constant_1#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> constant_2#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> constant_3#0: NoAlias

// CHECK-DAG: constant_1#0 <-> constant_2#0: MayAlias
// CHECK-DAG: constant_1#0 <-> constant_3#0: MayAlias
// CHECK-DAG: constant_1#0 <-> func.region0#0: MayAlias

// CHECK-DAG: constant_2#0 <-> constant_3#0: MayAlias
// CHECK-DAG: constant_2#0 <-> func.region0#0: MayAlias

// CHECK-DAG: constant_3#0 <-> func.region0#0: MayAlias

module {

func.func @constants(%arg: memref<2xf32>) attributes {test.ptr = "func"} {
  %1 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>

  %c0 = arith.constant {test.ptr = "constant_1"} 0 : index
  %c0_2 = arith.constant {test.ptr = "constant_2"} 0 : index
  %c1 = arith.constant {test.ptr = "constant_3"} 1 : index

  return
}

}
