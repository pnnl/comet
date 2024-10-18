// RUN: comet-opt %s --mem-access-pattern-analysis | FileCheck %s

module {
func.func @random_access() {
    %A = memref.alloc() : memref<100xi32>
    %B = memref.alloc() : memref<100xi32>

    //Random access due to the modulo operation
    affine.for %j = 0 to 50 {
      %v1 = affine.load %A[%j * 5 mod 37] : memref<100xi32>
      affine.store %v1, %B[%j] : memref<100xi32>
    }
    return
  }
}

// CHECK-DAG: Random access detected for memory location: %alloc = memref.alloc() : memref<100xi32>
// CHECK-DAG: Sequential access detected for memory location: %alloc_0 = memref.alloc() : memref<100xi32>