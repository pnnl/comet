// RUN: comet-opt %s --mem-access-pattern-analysis | FileCheck %s

module {
func.func @sequential_access() {
    %A = memref.alloc() : memref<10xi32>
    %B = memref.alloc() : memref<10xi32>

    // Sequential access
    affine.for %i = 0 to 10 {
      %v0 = affine.load %A[%i] : memref<10xi32>
      affine.store %v0, %B[%i] : memref<10xi32>
    }
    return
  }
}

// CHECK-DAG: Sequential access detected for memory location: %alloc = memref.alloc() : memref<10xi32>
// CHECK-DAG: Sequential access detected for memory location: %alloc_0 = memref.alloc() : memref<10xi32>