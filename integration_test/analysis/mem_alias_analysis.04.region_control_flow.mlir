// RUN: comet-opt %s --mem-alias-analysis 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "region_control_flow"
// CHECK-DAG: alloca_1#0 <-> if_alloca#0: MustAlias
// CHECK-DAG: alloca_1#0 <-> if_alloca_merge#0: MayAlias
// CHECK-DAG: alloca_1#0 <-> if_alloc#0: NoAlias

// CHECK-DAG: alloca_2#0 <-> if_alloca#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> if_alloca_merge#0: MayAlias
// CHECK-DAG: alloca_2#0 <-> if_alloc#0: NoAlias

// CHECK-DAG: alloc_1#0 <-> if_alloca#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> if_alloca_merge#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> if_alloc#0: MustAlias

// CHECK-DAG: if_alloca#0 <-> if_alloca_merge#0: MayAlias
// CHECK-DAG: if_alloca#0 <-> if_alloc#0: NoAlias
// CHECK-DAG: if_alloca#0 <-> func.region0#0: NoAlias
// CHECK-DAG: if_alloca#0 <-> func.region0#1: NoAlias

// CHECK-DAG: if_alloca_merge#0 <-> if_alloc#0: NoAlias
// CHECK-DAG: if_alloca_merge#0 <-> func.region0#0: NoAlias
// CHECK-DAG: if_alloca_merge#0 <-> func.region0#1: NoAlias

// CHECK-DAG: if_alloc#0 <-> func.region0#0: NoAlias
// CHECK-DAG: if_alloc#0 <-> func.region0#1: NoAlias

module {

func.func @region_control_flow(%arg: memref<2xf32>, %cond: i1) attributes {test.ptr = "func"} {
  %0 = memref.alloca() {test.ptr = "alloca_1"} : memref<8x64xf32>
  %1 = memref.alloca() {test.ptr = "alloca_2"} : memref<8x64xf32>
  %2 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>

  %3 = scf.if %cond -> (memref<8x64xf32>) {
    scf.yield %0 : memref<8x64xf32>
  } else {
    scf.yield %0 : memref<8x64xf32>
  } {test.ptr = "if_alloca"}

  %4 = scf.if %cond -> (memref<8x64xf32>) {
    scf.yield %0 : memref<8x64xf32>
  } else {
    scf.yield %1 : memref<8x64xf32>
  } {test.ptr = "if_alloca_merge"}

  %5 = scf.if %cond -> (memref<8x64xf32>) {
    scf.yield %2 : memref<8x64xf32>
  } else {
    scf.yield %2 : memref<8x64xf32>
  } {test.ptr = "if_alloc"}
  return
}

}