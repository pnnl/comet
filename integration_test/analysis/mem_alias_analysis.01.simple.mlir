// RUN: comet-opt %s --mem-alias-analysis 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "simple"
// CHECK-DAG: func.region0#0 <-> func.region0#1: MayAlias

// CHECK-DAG: alloca_1#0 <-> alloca_2#0: NoAlias
// CHECK-DAG: alloca_1#0 <-> alloc_1#0: NoAlias
// CHECK-DAG: alloca_1#0 <-> alloc_2#0: NoAlias
// CHECK-DAG: alloca_1#0 <-> func.region0#0: NoAlias
// CHECK-DAG: alloca_1#0 <-> func.region0#1: NoAlias

// CHECK-DAG: alloca_2#0 <-> alloc_1#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> alloc_2#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> func.region0#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> func.region0#1: NoAlias

// CHECK-DAG: alloc_1#0 <-> alloc_2#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> func.region0#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> func.region0#1: NoAlias

// CHECK-DAG: alloc_2#0 <-> func.region0#0: NoAlias
// CHECK-DAG: alloc_2#0 <-> func.region0#1: NoAlias
module{

func.func @simple(%arg: memref<2xf32>, %arg1: memref<2xf32>) attributes {test.ptr = "func"} {
  %0 = memref.alloca() {test.ptr = "alloca_1"} : memref<8x64xf32>
  %1 = memref.alloca() {test.ptr = "alloca_2"} : memref<8x64xf32>
  %2 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>
  %3 = memref.alloc() {test.ptr = "alloc_2"} : memref<8x64xf32>
  return
}

}
