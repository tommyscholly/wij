use std::fmt::Display;

use crate::parse::{Token, lex::Keyword};

use super::{ParseError, ParseErrorKind, Parseable, Parser, Spanned, typed::FunctionSignature};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    Int,
    Bool,
    Str,
    Byte,
    Ptr(Box<Type>),
    Array(Box<Type>, usize),
    Tuple(Vec<Type>),
    Fn(Box<FunctionSignature>),
    UserDef(String),
    Record(Vec<(String, Type)>),
    Unit,
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Type::{Array, Bool, Byte, Fn, Int, Ptr, Record, Str, Tuple, Unit, UserDef};
        match self {
            Int => write!(f, "int"),
            Bool => write!(f, "bool"),
            Str => write!(f, "str"),
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
            Ptr(t) => write!(f, "*{}", t),
            Byte => write!(f, "byte"),
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
            Some((Token::Star, span)) => {
                parser.pop_next();
                let (inner_ty, span_end) = Type::parse(parser)?;
                let span = span.start..span_end.end;
                Ok((Type::Ptr(Box::new(inner_ty)), span))
            }
            Some((Token::LBracket, span)) => {
                parser.pop_next();
                let (ty, span_end) = Type::parse(parser)?;
                let _ = parser.expect_next(Token::RBracket)?;
                let span = span.start..span_end.end;
                Ok((Type::Array(Box::new(ty), 0), span))
            }
            _ => {
                let (kw, span) = parser.expect_kw()?;
                match kw {
                    Keyword::Int => Ok((Type::Int, span)),
                    Keyword::Bool => Ok((Type::Bool, span)),
                    Keyword::Str => Ok((Type::Str, span)),
                    Keyword::Byte => Ok((Type::Byte, span)),
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
