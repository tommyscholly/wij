mod ast;
mod compiler;
mod comptime;
mod lex;
mod mir;

pub use ast::{
    ParseError, Parser,
    typed::{DeclKind, Module, ScopedCtx, TypeChecker},
    use_analysis,
};
pub use compiler::{
    Compiler, CompilerError, CompilerErrorKind, CompilerResult, ExportedSymbol, Import, ModuleInfo,
    SymbolKind,
};
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
