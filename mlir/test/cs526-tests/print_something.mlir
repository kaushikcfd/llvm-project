llvm.mlir.global constant @str("Hello World!\0A\00") : !llvm<"[14 x i8]">
llvm.func @printf(!llvm<"i8*">, ...) -> !llvm.i32

func @print_something() {
  %0 = llvm.mlir.addressof @str : !llvm<"[14 x i8]*">
  %1 = llvm.mlir.constant(0: i32) : !llvm.i32

  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %2 = llvm.getelementptr %0[%1, %1] : (!llvm<"[14 x i8]*">, !llvm.i32,
          !llvm.i32) -> !llvm<"i8*">
      llvm.call @printf(%2) : (!llvm<"i8*">) -> !llvm.i32
    }
  }
  return
}

func @main() {
  call @print_something() : ()->()
  return
}
