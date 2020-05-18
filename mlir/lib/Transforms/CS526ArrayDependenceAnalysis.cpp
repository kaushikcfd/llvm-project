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
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
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



static int64_t computeGCD(int64_t n1, int64_t n2) {
    if (n2 != 0)
        return computeGCD(n2, n1 % n2);
    else
        return n1;
}


static int64_t computeGCD(SmallVectorImpl<int64_t>* vec) {
  assert(vec->size() > 0);

  int64_t gcd = (*vec)[0];
  for (unsigned i=1; i < vec->size(); ++i)
    gcd = computeGCD(gcd, (*vec)[i]);

  return gcd;
}

// {{{ ZIV

static short zivCompareAffineMaps(AffineMap map1, AffineMap map2) {
  assert(map1.getNumResults() == map2.getNumResults());

  SmallVector<int64_t, 2> map1Indices, map2Indices;

  for (AffineExpr expr : map1.getResults()) {
    if (auto constAffine = expr.dyn_cast<AffineConstantExpr>())
      map1Indices.push_back(constAffine.getValue());
    else
      return 0;
  }

  int iDim = 0;

  for (AffineExpr expr : map2.getResults()) {
    if (auto constAffine = expr.dyn_cast<AffineConstantExpr>()) {
      if (constAffine.getValue() != map1Indices[iDim++])
        return -1;
    }
    else
      return 0;
  }

  return 1;
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

  auto getMemref = [](Operation* Op) {
    if (auto load = dyn_cast<AffineLoadOp>(Op))
      return load.getMemRef();
    auto store = dyn_cast<AffineStoreOp>(Op);
    assert(store && "Op should be either load or store");
    return store.getMemRef();
  };

  auto getAffineMap = [](Operation* Op) {
    if (auto load = dyn_cast<AffineLoadOp>(Op))
      return load.getAffineMap();
    auto store = dyn_cast<AffineStoreOp>(Op);
    assert(store && "Op should be either load or store");
    return store.getAffineMap();
  };

  for (int i=0; i < N; ++i) {
    Operation* OpI = affineLoadsStores[i];
    Value memrefI = getMemref(OpI);
    AffineMap affineMapI = getAffineMap(OpI);
    for (int j=0; j < N; ++j) {
      Operation* OpJ = affineLoadsStores[j];
      Value memrefJ = getMemref(OpJ);
      AffineMap affineMapJ = getAffineMap(OpJ);

      if (memrefI == memrefJ)
        R[i][j] = zivCompareAffineMaps(affineMapI, affineMapJ);
      else
        R[i][j] = -1;
    }
  }
  
  printMatrix(R, affineLoadsStores);
  llvm::dbgs() << "\n\n";
}

// }}}



/*
 * Returns:
 *  1  if map1 and map2 have an intersection according to GCD test.
 * -1  if map1 and map2 do not have an intersection according to GCD test.
 *  0  if GCD test fails.
 */
static short gcdTestCompareAffineMaps(AffineMap map1, AffineMap map2) {
  assert(map1.getNumResults() == map2.getNumResults());

  int NResults = map1.getNumResults();

  /// affineCoeffs[i]: contains all the linear indices for the i-th result
  /// dimension
  /// For example:
  /// A[2*i+3*j +4, 5*i + 7*j, 8] will have affineCoeffs = [[2, 3], [5, 7], []]
  SmallVector<SmallVector<int64_t, 2>, 3> affineCoeffs(NResults);

  /// b0minusa0 contains the difference of the constant terms between map1 and
  /// map2.
  /// A[2*i+3*j +4, 5*i + 7*j, 8], B[3*i+7, 8, 2*j+9] will have b0minusa0 as
  /// [3, 8, 1]
  SmallVector<int64_t, 3> b0minusa0Vec(NResults, 0);


  for(int iResult=0; iResult < NResults; ++iResult) {
    AffineExpr map1Result = map1.getResult(iResult);
    AffineExpr map2Result = map2.getResult(iResult);

    SmallVector<int64_t, 3> map1ResultFlattenedExpr, map2ResultFlattenedExpr;

    if (failed(
          getFlattenedAffineExpr(
            map1Result, map1.getNumDims(), map1.getNumSymbols(), &map1ResultFlattenedExpr))) {
      llvm::dbgs() << "Failed expression flattening for " << map1Result << "\n";
      return 0;
    }

    if (failed(
          getFlattenedAffineExpr(
            map2Result, map2.getNumDims(), map2.getNumSymbols(), &map2ResultFlattenedExpr))) {
      llvm::dbgs() << "Failed expression flattening for " << map2Result << "\n";
      return 0;
    }

    for (unsigned iDim=0; iDim < map1.getNumDims(); ++iDim) {
      int64_t affineCoeff = map1ResultFlattenedExpr[iDim];
      if (affineCoeff != 0)
        affineCoeffs[iResult].push_back(abs(affineCoeff));
    }

    for (unsigned iDim=0; iDim < map2.getNumDims(); ++iDim) {
      int64_t affineCoeff = map2ResultFlattenedExpr[iDim];
      if (affineCoeff != 0)
        affineCoeffs[iResult].push_back(abs(affineCoeff));
    }

    b0minusa0Vec[iResult] = abs(
        map1ResultFlattenedExpr[map1.getNumDims()] - map2ResultFlattenedExpr[map2.getNumDims()]);
  }

  LLVM_DEBUG(
    llvm::dbgs() << "Performing GCD test of '" << map1 << "' and '" << map2 << "'.\n";

    llvm::dbgs() << "Need to take gcd of the list: [";
    for (auto k1 : affineCoeffs) {
      llvm::dbgs() << "[";
      for (auto k2 : k1) {
        llvm::dbgs() << k2 << ", ";
      }
      llvm::dbgs() << "], ";
    }

    llvm::dbgs() << "]\n";

    llvm::dbgs() << "b0 - a0 = [";
    for (auto k1 : b0minusa0Vec) {
      llvm::dbgs() << k1 << ", ";
    }

    llvm::dbgs() << "]\n";);


  int iResult = 0;

  for(auto affineCoeff : affineCoeffs) {
    int64_t b0Minusa0 = b0minusa0Vec[iResult++];
    if (affineCoeff.size() == 0) {
      // ZIV
      if (b0Minusa0 != 0) {
        LLVM_DEBUG(llvm::dbgs() << "GCD Test returned -1 for a ZIV.\n");
        return -1;
      }
      continue;
    }
    // MIV
    int64_t gcd = computeGCD(&affineCoeff);
    if ((b0Minusa0 % gcd) != 0) {
      LLVM_DEBUG(llvm::dbgs() << "GCD Test returned -1.\n");
      return -1;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "GCD Test returned 1.\n");

  return 1;
}


/*
 * Implements the GCD test.
 */

void printGCDTestResults(std::vector<Operation*> affineLoadsStores) {
  int N = affineLoadsStores.size();
  auto R = createMatrix<short>(N);

  auto getMemref = [](Operation* Op) {
    if (auto load = dyn_cast<AffineLoadOp>(Op))
      return load.getMemRef();
    auto store = dyn_cast<AffineStoreOp>(Op);
    assert(store && "Op should be either load or store");
    return store.getMemRef();
  };

  auto getAffineMap = [](Operation* Op) {
    if (auto load = dyn_cast<AffineLoadOp>(Op))
      return load.getAffineMap();
    auto store = dyn_cast<AffineStoreOp>(Op);
    assert(store && "Op should be either load or store");
    return store.getAffineMap();
  };

  for (int i=0; i < N; ++i) {
    Operation* OpI = affineLoadsStores[i];
    Value memrefI = getMemref(OpI);
    AffineMap affineMapI = getAffineMap(OpI);
    for (int j=0; j < N; ++j) {
      Operation* OpJ = affineLoadsStores[j];
      Value memrefJ = getMemref(OpJ);
      AffineMap affineMapJ = getAffineMap(OpJ);

      if (memrefI == memrefJ)
        R[i][j] = gcdTestCompareAffineMaps(affineMapI, affineMapJ);
      else
        // TODO: assumming memrefs do not alias each other.
        R[i][j] = -1;
    }
  }

  printMatrix(R, affineLoadsStores);
  llvm::dbgs() << "\n\n";
}



void CS526ArrayDependenceAnalysis::runOnFunction() {
  FuncOp func = getFunction();

  std::vector<Operation*> affineLoadsStores;

  // populate affineLoadsStores
  func.walk([&](Operation *op) {
    if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)) {
      affineLoadsStores.push_back(op);
    }
  });

  // printZIVResults(affineLoadsStores);
  printGCDTestResults(affineLoadsStores);
}


// Register this pass to make it accessible to utilities like mlir-opt.
// (Pass registration is discussed more below)
PassRegistration<CS526ArrayDependenceAnalysis> pass(
      "cs526-array-dep-analysis",
      "[CS526]: Check dependencies of all memrefs");

// vim:foldmethod=marker
