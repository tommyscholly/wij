mod ast;
mod lex;
mod mir;

pub use ast::{
    ParseError, Parser,
    typed::{Module, ScopedCtx, type_check},
    use_analysis,
};
pub use lex::tokenize;
pub use mir::ssa::build_ssa;

pub trait AstError {
    fn span(&self) -> Option<ast::Span>;
    fn reason(&self) -> String;
    fn notes(&self) -> Vec<(String, ast::Span)>;
}
