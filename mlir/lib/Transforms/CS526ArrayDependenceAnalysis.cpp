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
struct CS526ArrayDependenceAnalysis : public PassWrapper<CS526ArrayDependenceAnalysis, FunctionPass> {
  void runOnFunction() override {
    //TODO: Insert something here...
    // I think getFunction() gets us the function..
    llvm::dbgs() << "Hello World!!\n";
  }

};
} // end anonymous namespace

// Register this pass to make it accessible to utilities like mlir-opt.
// (Pass registration is discussed more below)
PassRegistration<CS526ArrayDependenceAnalysis> pass(
      "cs526-array-dep-analysis",
      "[CS526]: Check dependencies of all memrefs");
