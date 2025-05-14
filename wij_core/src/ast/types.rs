use std::fmt::Display;

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

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Type::{Array, Bool, Fn, Int, Record, Tuple, Unit, UserDef};
        match self {
            Int => write!(f, "int"),
            Bool => write!(f, "bool"),
            Type::String => write!(f, "str"),
            Array(t, len) => write!(f, "[{}; {}]", t, len),
            Tuple(types) => write!(
                f,
                "({})",
                types
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Fn(sig) => write!(f, "Fn({})", sig),
            UserDef(ident) => write!(f, "{}", ident),
            Record(fields) => write!(
                f,
                "Record({})",
                fields
                    .iter()
                    .map(|(name, ty)| format!("{}: {}", name, ty))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Unit => write!(f, "()"),
        }
    }
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
