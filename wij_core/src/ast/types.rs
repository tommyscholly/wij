use crate::parse::{Token, lex::Keyword};

use super::{ParseError, ParseErrorKind, Parseable, Parser, Spanned, typed::FunctionSignature};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    Int,
    Bool,
    String,
    Array(Box<Type>, usize),
    Tuple(Vec<Type>),
    Fn(Box<FunctionSignature>),
    UserDef(String),
    Record(Vec<(String, Type)>),
    Unit,
}

impl Parseable for Type {
    fn parse(parser: &mut Parser) -> Result<Spanned<Self>, ParseError> {
        match parser.peek_next() {
            Some((Token::Identifier(ident), span)) => {
                parser.pop_next();
                Ok((Type::UserDef(ident), span))
            }
            _ => {
                let (kw, span) = parser.expect_kw()?;
                match kw {
                    Keyword::Int => Ok((Type::Int, span)),
                    Keyword::Bool => Ok((Type::Bool, span)),
                    _ => Err(ParseError::with_reason(
                        ParseErrorKind::MalformedType,
                        span,
                        &format!("Expected type, got {:?}", kw),
                    )),
                }
            }
        }
    }
}
