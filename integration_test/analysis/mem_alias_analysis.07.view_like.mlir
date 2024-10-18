// RUN: comet-opt %s --mem-alias-analysis 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "view_like"
// CHECK-DAG: alloc_1#0 <-> view#0: NoAlias

// CHECK-DAG: alloca_1#0 <-> view#0: MustAlias

// CHECK-DAG: view#0 <-> func.region0#0: NoAlias
// CHECK-DAG: view#0 <-> func.region0#1: NoAlias

module {

func.func @view_like(%arg: memref<2xf32>, %size: index) attributes {test.ptr = "func"} {
  %1 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>

  %c0 = arith.constant 0 : index
  %2 = memref.alloca (%size) {test.ptr = "alloca_1"} : memref<?xi8>
  %3 = memref.view %2[%c0][] {test.ptr = "view"} : memref<?xi8> to memref<8x64xf32>
  return
}

}
