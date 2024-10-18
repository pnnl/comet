// RUN: comet-opt %s --mem-alias-analysis 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "control_flow"
// CHECK-DAG: alloca_1#0 <-> func.region0.block1#0: MustAlias
// CHECK-DAG: alloca_1#0 <-> func.region0.block2#0: MustAlias

// CHECK-DAG: alloca_2#0 <-> func.region0.block1#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> func.region0.block2#0: NoAlias

// CHECK-DAG: alloc_1#0 <-> func.region0.block1#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> func.region0.block2#0: NoAlias

// CHECK-DAG: func.region0#0 <-> func.region0.block1#0: NoAlias
// CHECK-DAG: func.region0#0 <-> func.region0.block2#0: NoAlias

// CHECK-DAG: func.region0#1 <-> func.region0.block1#0: NoAlias
// CHECK-DAG: func.region0#1 <-> func.region0.block2#0: NoAlias

// CHECK-DAG: func.region0.block1#0 <-> func.region0.block2#0: MustAlias
module {

func.func @control_flow(%arg: memref<2xf32>, %cond: i1) attributes {test.ptr = "func"} {
  %0 = memref.alloca() {test.ptr = "alloca_1"} : memref<8x64xf32>
  %1 = memref.alloca() {test.ptr = "alloca_2"} : memref<8x64xf32>
  %2 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>

  cf.cond_br %cond, ^bb1(%0 : memref<8x64xf32>), ^bb2(%0 : memref<8x64xf32>)

^bb1(%arg1: memref<8x64xf32>):
  cf.br ^bb2(%arg1 : memref<8x64xf32>)

^bb2(%arg2: memref<8x64xf32>):
  return
}

}
