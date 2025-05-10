mod ast;
mod parse;

pub use ast::{ParseError, Parser};
pub use parse::lex::tokenize;

pub trait AstError {
    fn span(&self) -> Option<ast::Span>;
    fn reason(&self) -> &str;
}
