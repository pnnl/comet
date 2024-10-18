// RUN: comet-opt %s --mem-access-frequency-analysis | FileCheck %s

module {
func.func @affine_map() {
    %A = memref.alloc() : memref<20xi32>
    %B = memref.alloc() : memref<10xi32>

    affine.for %i = 0 to 10 {
        %index = affine.apply affine_map<(d0) -> (2 * d0 + 1)> (%i)
        %v0 = affine.load %A[%index] : memref<20xi32>
        affine.if affine_set<(d0) : (d0 mod 2 == 0) > (%i) {
            affine.store %v0, %B[%i] : memref<10xi32>
        }
    }
    return
  }
}

// CHECK-DAG: Memory location %alloc_0 = memref.alloc() : memref<10xi32> was accessed 6 times.
// CHECK-DAG: Memory location %alloc = memref.alloc() : memref<20xi32> was accessed 11 times.