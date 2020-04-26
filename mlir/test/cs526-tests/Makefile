%.ll: %.mlir
	mlir-opt -lower-affine -convert-loop-to-std -convert-std-to-llvm $< | mlir-translate --mlir-to-llvmir -o $@
