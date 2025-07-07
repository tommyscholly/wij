pub mod ast;
pub mod compiler;
mod comptime;
mod lex;
mod mir;

pub use ast::{
    Import, ParseError, Parser,
    typed::{DeclKind, Module, ScopedCtx, TypeChecker, TypedDecl},
    use_analysis,
};
pub use compiler::{Compiler, CompilerError, CompilerResult, ModuleInfo};
pub use lex::tokenize;
pub use mir::ssa::{self, Program, build_ssa};

pub type Span = std::ops::Range<usize>;

pub trait WijError {
    fn span(&self) -> Option<Span>;
    fn reason(&self) -> String;
    fn notes(&self) -> Vec<(String, Span)>;
}

pub trait Graphviz {
    fn dot(&self) -> String;
}

pub trait SizeOf {
    fn size_of(&self) -> usize;
}
