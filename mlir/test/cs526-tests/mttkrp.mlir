// Inspired by http://tensor-compiler.org/docs/data_analytics/
// A[i,j] = B[i,k,l] * D[l,j] * C[k,j]

func @mttkrp(%A: memref<128x256xf64>, %B: memref<128x64x32xf64>, %C: memref<64x256xf64>, %D: memref<32x256xf64>) {
  %0 = constant 0.0 : f64
  affine.for %i = 0 to 128 {
    affine.for %j = 0 to 256 {
      %outer = alloc() : memref<1xf64>
      affine.store %0, %outer[0] : memref<1xf64>

      affine.for %k = 0 to 64 {
        %inner = alloc() : memref<1xf64>
        affine.store %0, %inner[0] : memref<1xf64>

        affine.for %l = 0 to 32 {
          %b = affine.load %B[%i, %k, %l] : memref<128x64x32xf64>
          %c = affine.load %C[%k, %j] : memref <64x256xf64>
          %d = affine.load %D[%l, %j] : memref <32x256xf64>
          %prod1 = mulf %b, %d : f64
          %prod2 = mulf %prod1, %c : f64

          %cur = affine.load %inner[0] : memref<1xf64>
          %new = addf %prod2, %cur : f64
          affine.store %new, %inner[0] : memref<1xf64>
        }

        %cur = affine.load %outer[0] : memref<1xf64>
        %inner_sum = affine.load %inner[0] : memref<1xf64>
        %new = addf %cur, %inner_sum : f64
        affine.store %new, %outer[0] : memref<1xf64>
      }

      %val = affine.load %outer[0] : memref<1xf64>
      affine.store %val, %A[%i,%j] : memref<128x256xf64>
    }
  }
  return
}
