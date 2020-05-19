// for _ <- 0..10
//   A[0] = B[0]

func @ziv2(%A: memref<10xf32>, %B: memref<10xf32>) {
  affine.for %i = 0 to 10 step 1 {
    %0 = affine.load %B[0] : memref<10xf32>
    affine.store %0, %A[0] : memref<10xf32>
  }
  return
}

// Expected output:
// |-----+----+----+
// |     | L6 | S7 |
// |-----+----+----+
// | L6  | T  | F  |
// | S7  |    | T  |
// |-----+----+----+
