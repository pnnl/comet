// RUN: comet-opt %s --mem-alias-analysis 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "region_loop_control_flow"
// CHECK-DAG: alloca_1#0 <-> for_alloca#0: MustAlias
// CHECK-DAG: alloca_1#0 <-> for_alloca.region0#0: MayAlias
// CHECK-DAG: alloca_1#0 <-> for_alloca.region0#1: MustAlias

// CHECK-DAG: alloca_2#0 <-> for_alloca#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> for_alloca.region0#0: MayAlias
// CHECK-DAG: alloca_2#0 <-> for_alloca.region0#1: NoAlias

// CHECK-DAG: alloc_1#0 <-> for_alloca#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> for_alloca.region0#0: MayAlias
// CHECK-DAG: alloc_1#0 <-> for_alloca.region0#1: NoAlias

// CHECK-DAG: for_alloca#0 <-> for_alloca.region0#0: MayAlias
// CHECK-DAG: for_alloca#0 <-> for_alloca.region0#1: MustAlias
// CHECK-DAG: for_alloca#0 <-> func.region0#0: NoAlias
// CHECK-DAG: for_alloca#0 <-> func.region0#1: NoAlias
// CHECK-DAG: for_alloca#0 <-> func.region0#2: NoAlias
// CHECK-DAG: for_alloca#0 <-> func.region0#3: NoAlias

// CHECK-DAG: for_alloca.region0#0 <-> for_alloca.region0#1: MayAlias
// CHECK-DAG: for_alloca.region0#0 <-> func.region0#0: MayAlias
// CHECK-DAG: for_alloca.region0#0 <-> func.region0#1: MayAlias
// CHECK-DAG: for_alloca.region0#0 <-> func.region0#2: MayAlias
// CHECK-DAG: for_alloca.region0#0 <-> func.region0#3: MayAlias

// CHECK-DAG: for_alloca.region0#1 <-> func.region0#0: NoAlias
// CHECK-DAG: for_alloca.region0#1 <-> func.region0#1: NoAlias
// CHECK-DAG: for_alloca.region0#1 <-> func.region0#2: NoAlias
// CHECK-DAG: for_alloca.region0#1 <-> func.region0#3: NoAlias

module {

func.func @region_loop_control_flow(%arg: memref<2xf32>, %loopI0 : index,
                               %loopI1 : index, %loopI2 : index) attributes {test.ptr = "func"} {
  %0 = memref.alloca() {test.ptr = "alloca_1"} : memref<8x64xf32>
  %1 = memref.alloca() {test.ptr = "alloca_2"} : memref<8x64xf32>
  %2 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>

  %result = scf.for %i0 = %loopI0 to %loopI1 step %loopI2 iter_args(%si = %0) -> (memref<8x64xf32>) {
    scf.yield %si : memref<8x64xf32>
  } {test.ptr = "for_alloca"}
  return
}

}