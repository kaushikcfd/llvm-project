// for _ <- 0..10
//   A[0] = B[0] + B[1]

func @ziv3(%A: memref<10xf32>, %B: memref<10xf32>) {
  affine.for %i = 0 to 10 step 1 {
    %0 = affine.load %B[0] : memref<10xf32>
    %1 = affine.load %B[1] : memref<10xf32>
    %2 = addf %0, %1 : f32
    affine.store %2, %A[0] : memref<10xf32>
  }
  return
}

// Expected output:
// |-----+----+----+----+
// |     | L6 | L7 | S8 | 
// |-----+----+----+----+
// | L6  | T  | F  | F  |
// | L7  |    | T  | F  |
// | S8  |    |    | T  |
// |-----+----+----+----+
