// for j in 1:N
//   for i in 1:N
//     X[i, j] = W[i, j] + 1 (S1)
//     Y[i, j] = X[100-i, j] (S2)
//   endfor
// endfor

func @siv2(%W: memref<100x100xf32>, %X: memref<100x100xf32>, %Y: memref<100x100xf32>, %N: index) {
  %0 = constant 1.0 : f32
  affine.for %j = 0 to %N {
    affine.for %i = 0 to %N {
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

