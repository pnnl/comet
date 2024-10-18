// RUN: comet-opt %s --mem-access-frequency-analysis | FileCheck %s
module {
func.func @affine_for_step() {
    %A = memref.alloc() : memref<10xi32>
    %B = memref.alloc() : memref<10xi32>

    affine.for %i = 0 to 10 step 2 {
      %v0 = affine.load %A[%i] : memref<10xi32>
      affine.if affine_set<(d0) : (d0 mod 2 == 0) > (%i) {
        affine.store %v0, %B[%i] : memref<10xi32>
      }
    }

    return
  }
}

// CHECK-DAG: Memory location %alloc = memref.alloc() : memref<10xi32> was accessed 6 times.
// CHECK-DAG: Memory location %alloc_0 = memref.alloc() : memref<10xi32> was accessed 3 times.

