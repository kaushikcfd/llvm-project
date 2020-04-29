// for j in 1:100
//   for i in 1:100
//     X[i, j] = W[i, j] + 1 (S1)
//     Y[i, j] = X[100-i, j] (S2)
//   endfor
// endfor

func @siv1(%W: memref<100x100xf32>, %X: memref<100x100xf32>, %Y: memref<100x100xf32>) {
  %0 = constant 1.0 : f32
  affine.for %j = 0 to 100 {
    affine.for %i = 0 to 100 {
      // S1:
      %Wij = affine.load %W[%i, %j] : memref<100x100xf32>
      %Xij = addf %Wij, %0: f32
      affine.store %Xij, %X[%i, %j] : memref<100x100xf32>

      // S2:
      %Yij = affine.load %X[100-%i, %j] : memref<100x100xf32>
      affine.store %Yij, %Y[%i, %j] : memref<100x100xf32>
    }
  }
  return
}

// Expected output:
//|-----+-----+-----+-----+-----|
//|     | L13 | S15 | L18 | S19 |
//|-----+-----+-----+-----+-----|
//| L13 |   1 |  -1 |  -1 |  -1 |
//| S15 |  -1 |   1 |   1 |  -1 |
//| L18 |  -1 |   1 |   1 |  -1 |
//| S19 |  -1 |  -1 |  -1 |   1 |
//|-----+-----+-----+-----+-----|
