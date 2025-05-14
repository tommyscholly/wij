mod ast;
mod parse;

pub use ast::{ParseError, Parser, typed::type_check};
pub use parse::lex::tokenize;

pub trait AstError {
    fn span(&self) -> Option<ast::Span>;
    fn reason(&self) -> String;
}
