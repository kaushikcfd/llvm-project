// for j in 10:90
//   for i in 10:90
//      X[i,j] = X[i+3,j-3]
//      X[i-6,j+9] = X[i-1,j-1]
//      Y[i+6,j+6] = X[i-4,j-2]

func @siv3(%X: memref<100x100xf32>, %Y: memref<100x100xf32>) {
  affine.for %j = 10 to 90 {
    affine.for %i = 10 to 90 {
      %x1 = affine.load %X[%i + 3, %j - 3] : memref<100x100xf32>
      affine.store %x1, %X[%i + 0, %j + 0] : memref<100x100xf32>

      %x2 = affine.load %X[%i - 1, %j - 1] : memref<100x100xf32>
      affine.store %x2, %X[%i - 6, %j + 9] : memref<100x100xf32>

      %x3 = affine.load %X[%i - 4, %j - 2] : memref<100x100xf32>
      affine.store %x3, %Y[%i + 6, %j + 6] : memref<100x100xf32>
    }
  }
  return
}

