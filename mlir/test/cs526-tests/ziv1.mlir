// A[3, 4] = 2.0
// B[4] = 2 * A[4, 5]
// C[1, 2, 3] = A[3, 4]
func @ziv1(%A: memref<10x10xf32>, %B: memref<10xf32>, %C: memref<10x10x10xf32>) {
  %0 = constant 2.0 : f32
  affine.store %0, %A[3, 4] : memref<10x10xf32>
  %1 = affine.load %A[4, 5] : memref<10x10xf32>
  %2 = mulf %0, %1 : f32
  affine.store %2, %B[4] : memref<10xf32>
  %3 = affine.load %A[3, 4] : memref<10x10xf32>
  affine.store %3, %C[1, 2, 3] : memref<10x10x10xf32>
  return
}

// Expected output:
// |-----+----+----+----+-----+-----|
// |     | S6 | L7 | S9 | L10 | S11 |
// |-----+----+----+----+-----+-----|
// | S6  | T  | F  | F  | T   | F   |
// | L7  |    | T  | F  | F   | F   |
// | S9  |    |    | T  | F   | F   |
// | L10 |    |    |    | T   | F   |
// | S11 |    |    |    |     | T   |
// |-----+----+----+----+-----+-----|
