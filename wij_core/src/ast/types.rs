use crate::parse::lex::Keyword;

use super::{ParseError, ParseErrorKind, Parseable, Parser, Spanned};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    Int,
    UserDef(String),
}

impl Parseable for Type {
    fn parse(parser: &mut Parser) -> Result<Spanned<Self>, ParseError> {
        match parser.expect_ident() {
            Ok((ident, span)) => Ok((Type::UserDef(ident), span)),
            Err(_) => {
                let (kw, span) = parser.expect_kw()?;
                match kw {
                    Keyword::Int => Ok((Type::Int, span)),
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
