func @saxpy(%a : f32, %x : memref<?xf32>, %y : memref<?xf32>) {
  %n = dim %x, 0 : memref<?xf32>
  // Assuming that %y is also of the same length.
  // In future maybe add an assert here?

  affine.for %i = 0 to %n {
    %xi = affine.load %x[%i] : memref<?xf32>
    %axi =  mulf %a, %xi : f32
    %yi = affine.load %y[%i] : memref<?xf32>
    %axpyi = addf %yi, %axi : f32
    affine.store %axpyi, %y[%i] : memref<?xf32>
  }
  return
}



func @daxpy(%a : f64, %x : memref<?xf64>, %y : memref<?xf64>) {
  %n = dim %x, 0 : memref<?xf64>
  // Assuming that %y is also of the same length.
  // In future maybe add an assert here?

  affine.for %i = 0 to %n {
    %xi = affine.load %x[%i] : memref<?xf64>
    %axi =  mulf %a, %xi : f64
    %yi = affine.load %y[%i] : memref<?xf64>
    %axpyi = addf %yi, %axi : f64
    affine.store %axpyi, %y[%i] : memref<?xf64>
  }
  return
}
