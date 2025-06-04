mod cranelift;
mod llvm;

use std::collections::HashMap;

use wij_core::{
    Program,
    ssa::{FnID, FunctionType},
};

pub enum Backend {
    Cranelift,
    Llvm,
    Mlir,
}

pub struct CodegenOptions {
    backend: Backend,
    program_name: String,
}

impl CodegenOptions {
    pub fn new(program_name: String, backend: Backend) -> Self {
        Self {
            backend,
            program_name,
        }
    }
}

pub fn codegen(program: Program, options: CodegenOptions) {
    match &options.backend {
        Backend::Cranelift => cranelift::compile(program, &options.program_name),
        Backend::Llvm => llvm::compile(),
        Backend::Mlir => todo!(),
    }
}
