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

// TODO: Add a statistic to report how many pairs were analyzed.

namespace {


template <typename T>
using Matrix = std::vector<std::vector<T>>;

struct CS526ArrayDependenceAnalysis : public PassWrapper<CS526ArrayDependenceAnalysis, FunctionPass> {
  void runOnFunction() override;
};
} // end anonymous namespace



template <typename T>
static Matrix<T> createMatrix(size_t N) {
  Matrix<T> result(N, std::vector<T>(N));
  return result;
}

static void printMatrix(Matrix<short> R, std::vector<Operation*> affineLoadsStores) {
  // @TODO: Assign names to rows columsn
  int N = affineLoadsStores.size();

  const int cellWidth = 6;
  auto formatToCell = [cellWidth](std::string s) {
    int diff = cellWidth - s.size();
    assert((diff >= 0) && "cellWidth not suitable");
    std::string bigPad(diff/2 + diff%2, ' ');
    std::string smallPad(diff/2, ' ');
    if (cellWidth % 2)
      return smallPad + s + bigPad;
    else
      return bigPad + s + smallPad;
  };

  auto accessName = [](Operation* Op) {
    std::string result;
    if (isa<AffineLoadOp>(Op))
      result = "L";
    else if (isa<AffineStoreOp>(Op))
      result = "S";
    else
      assert(false && "Received a non-memory op!");

    if (auto fileLineCol = (Op->getLoc()).dyn_cast<FileLineColLoc>())
      result += std::to_string(fileLineCol.getLine());
    else
      result += "xx";

    return result;
  };

  // {{{ table begin (separator)
  
  llvm::dbgs() << "+";
  for (int i = -1; i < N; ++i) {
    llvm::dbgs() << formatToCell(std::string(cellWidth, '-'));
    llvm::dbgs() << "+";
  }
  llvm::dbgs() << "\n";

  // }}}

  // {{{ first header row

  llvm::dbgs() << "|";
  llvm::dbgs() << formatToCell("");
  llvm::dbgs() << "|";

  for (Operation* Op : affineLoadsStores) {
    llvm::dbgs() << formatToCell(accessName(Op));
    llvm::dbgs() << "|";
  }
  llvm::dbgs() << "\n";

  // }}}
  
  // {{{ second row (separator)
  
  llvm::dbgs() << "+";
  for (int i = -1; i < N; ++i) {
    llvm::dbgs() << formatToCell(std::string(cellWidth, '-'));
    llvm::dbgs() << "+";
  }
  llvm::dbgs() << "\n";

  // }}}

  // {{{ Print the matrix
  
  for (int i = 0; i < N; ++i)
  {
    llvm::dbgs() << "|";
    llvm::dbgs() << formatToCell(accessName(affineLoadsStores[i]));
    llvm::dbgs() << "|";
    for (int j = 0; j < N; ++j) {
      llvm::dbgs() << formatToCell(std::to_string(R[i][j]));
      llvm::dbgs() << "|";
    }
    llvm::dbgs() << "\n";
  }

  // }}}

  // {{{ last row (separator)
  
  llvm::dbgs() << "+";
  for (int i = -1; i < N; ++i) {
    llvm::dbgs() << formatToCell(std::string(cellWidth, '-'));
    llvm::dbgs() << "+";
  }
  llvm::dbgs() << "\n";

  // }}}
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
void printZIVResults(std::vector<Operation*> affineLoadsStores) {
  int N = affineLoadsStores.size();
  auto R = createMatrix<short>(N);
  // TODO: compute the dependencies here...
  printMatrix(R, affineLoadsStores);
}


void CS526ArrayDependenceAnalysis::runOnFunction() {
  FuncOp func = getFunction();
  llvm::dbgs() << "================================================================\n";
  llvm::dbgs() << "And the function of interest is -- \n";
  func.print(llvm::dbgs());
  llvm::dbgs() << "\n";
  llvm::dbgs() << "================================================================\n";

  std::vector<Operation*> affineLoadsStores;

  // populate affineLoadsStores
  func.walk([&](Operation *op) {
    if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) {
      affineLoadsStores.push_back(op);
    }
  });

  llvm::dbgs() << "Pretty printing out table:\n";
  printZIVResults(affineLoadsStores);
  llvm::dbgs() << "================================================================\n";
}


// Register this pass to make it accessible to utilities like mlir-opt.
// (Pass registration is discussed more below)
PassRegistration<CS526ArrayDependenceAnalysis> pass(
      "cs526-array-dep-analysis",
      "[CS526]: Check dependencies of all memrefs");
