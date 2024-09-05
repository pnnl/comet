// RUN: comet-opt %s --mem-access-frequency-analysis | FileCheck %s
module {

func.func @affine_for() {
    %A = memref.alloc() : memref<10xi32>
    %B = memref.alloc() : memref<10xi32>

    affine.for %i = 0 to 10 {
      %v0 = affine.load %A[%i] : memref<10xi32>
      %v1 = affine.load %B[%i] : memref<10xi32>
      affine.store %v0, %B[%i] : memref<10xi32>
    }

    return
  }
}

// CHECK: Memory location %alloc_0 = memref.alloc() : memref<10xi32> was accessed 22 times.
// CHECK: Memory location %alloc = memref.alloc() : memref<10xi32> was accessed 11 times.
