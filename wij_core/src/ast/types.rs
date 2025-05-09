use crate::parse::{Token, lex::Keyword};

use super::{ParseError, ParseErrorKind, Parseable, Parser, Spanned};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    Int,
}

impl Parseable for Type {
    fn parse(parser: &mut Parser) -> Result<Spanned<Self>, ParseError> {
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
