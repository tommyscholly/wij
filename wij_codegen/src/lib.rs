mod cranelift;
mod llvm;

use wij_core::Program;

pub enum Backend {
    Cranelift,
    Llvm,
    Mlir,
}

pub struct CodegenOptions {
    backend: Backend,
    program_name: String,
}

pub fn codegen(program: Program, options: CodegenOptions) {
    match &options.backend {
        Backend::Cranelift => cranelift::compile(program, &options.program_name),
        Backend::Llvm => llvm::compile(),
        Backend::Mlir => todo!(),
    }
}
