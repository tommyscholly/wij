pub mod lex;

mod macros;

pub use lex::{LexError, Lexer, Token};

#[cfg(test)]
mod tests;
