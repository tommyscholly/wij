mod ast;
mod lex;
mod mir;

pub use ast::{
    ParseError, Parser,
    typed::{Module, type_check},
    use_analysis,
};
pub use lex::tokenize;

pub trait AstError {
    fn span(&self) -> Option<ast::Span>;
    fn reason(&self) -> String;
    fn notes(&self) -> Vec<(String, ast::Span)>;
}
