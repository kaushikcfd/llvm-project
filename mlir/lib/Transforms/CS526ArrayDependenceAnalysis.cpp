//===- CS526ArrayDependenceAnalysis.cpp -Array dep analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to run pair-wise memref access dependence checks.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cs526-array-dep-analysis"

using namespace mlir;


namespace {


template <typename T>
using Matrix = vector<vector<T>>;


template <typename T>
static Matrix<T> createMatrix(size_t N) {
  Matrix<T> result(N, vector<T>(N));
  return result;
}




static void printMatrix(Matrix<short> R, vector<Op> affineLoadsStores) {
  // @TODO: Assign names to rows columsn
  return;
}


/*
 * Implments the ZIV test for a function.
 *
 * Walks through all the given loads and stores, and checks for dependencies
 * for each pair.
 *
 * Prints a dense matrix 'R' where:
 * 
 * Rij =  1    if Opi and Opj access the same memory
 * Rij = -1    if Opi and Opj do not access the same memory
 * Rij =  0    if we are unable to ascertain the above 2 conditions on the
 *             memory accessed by Opi, Opj
 */
void printZIVResults(vector<Op> affineLoadsStores) {
  auto R = createMatrix<short>(affineLoadsStores.size());
  printMatrix(R);
}



struct CS526ArrayDependenceAnalysis : public PassWrapper<CS526ArrayDependenceAnalysis, FunctionPass> {
  void runOnFunction() override {

    FuncOp func = getFunction();
    llvm::dbgs() << "================================================================\n";
    llvm::dbgs() << "And the function of interest is -- \n";
    func.print(llvm::dbgs());
    llvm::dbgs() << "\n";
    llvm::dbgs() << "================================================================\n";

    llvm::dbgs() << "Printing all the loads and stores in the functions:\n";

    func.walk([&](Operation *op) {
      if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) {
        op->print(llvm::dbgs());
        Location loc = op->getLoc();
        if (auto fileLineCol = loc.dyn_cast<FileLineColLoc>()) {
          llvm::dbgs() << " ; On line " << fileLineCol.getLine();
        }
        else {
          lvm::dbgs() << "Unkown loc.";
        }
        llvm::dbgs() << "\n";
      }
    });
    llvm::dbgs() << "================================================================\n";

  }

};
} // end anonymous namespace

// Register this pass to make it accessible to utilities like mlir-opt.
// (Pass registration is discussed more below)
PassRegistration<CS526ArrayDependenceAnalysis> pass(
      "cs526-array-dep-analysis",
      "[CS526]: Check dependencies of all memrefs");
