// Y = X*F
// X : <H+K-1 x W+K-1 xf32>
// F : <KxKxf32>
// Y : <HxWxf32>

func @conv2d(%X: memref<?x?xf32>, %F: memref<?x?xf32>, %Y: memref<?x?xf32>, %H: index, %W: index, %K: index) {
  %0 = constant 0.0 : f32
  affine.for %h = 0 to %H {
    affine.for %w = 0 to %W {
      %acc = alloc() : memref<1xf32>
      affine.store %0, %acc[0] : memref<1xf32>

      affine.for %p = 0 to %K {
        affine.for %q = 0 to %K {
          %x = affine.load %X[%h+%p, %w+%q] : memref<?x?xf32>
          %f = affine.load %F[%p, %q] : memref<?x?xf32>
          %prod = mulf %x, %f : f32

          %cur = affine.load %acc[0] : memref<1xf32>
          %new = addf %prod, %cur : f32
          affine.store %new, %acc[0] : memref<1xf32>
        }
      }

      %val = affine.load %acc[0] : memref<1xf32>
      affine.store %val, %Y[%h,%w] : memref<?x?xf32>
    }
  }

  return
}
