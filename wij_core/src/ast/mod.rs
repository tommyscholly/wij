mod types;

use crate::parse::{Token, lex::Keyword};
use types::Type;

use std::collections::VecDeque;

pub type Span = std::ops::Range<usize>;
pub type Spanned<T> = (T, Span);

#[derive(Debug)]
pub enum ParseErrorKind {
    MalformedVariable,
    MalformedExpression,
    MalformedType,
    MalformedLet,
    ExpectedSemiColon,
    EndOfInput,
}

#[derive(Debug)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Option<Span>,
    pub reason: Option<String>,
}

impl ParseError {
    pub fn with_reason(kind: ParseErrorKind, span: Span, reason: &str) -> Self {
        ParseError {
            kind,
            span: Some(span),
            reason: Some(reason.to_string()),
        }
    }

    pub fn new(kind: ParseErrorKind, span: Span) -> Self {
        ParseError {
            kind,
            span: Some(span),
            reason: None,
        }
    }

    pub fn spanless(kind: ParseErrorKind) -> Self {
        ParseError {
            kind,
            span: None,
            reason: None,
        }
    }
}

pub trait Parseable
where
    Self: Sized,
{
    fn parse(parser: &mut Parser) -> Result<Spanned<Self>, ParseError>;
}

pub struct Var {
    pub name: String,
    pub ty: Type,
}

impl Parseable for Var {
    fn parse(parser: &mut Parser) -> Result<Spanned<Self>, ParseError> {
        let (name, span) = match parser.pop_next() {
            Some((Token::Identifier(name), span)) => (name, span),
            Some(t) => {
                let span = t.1.clone();
                parser.push_front(t);
                return Err(ParseError::with_reason(
                    ParseErrorKind::MalformedVariable,
                    span,
                    "Expected variable name",
                ));
            }
            None => return Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        };

        let Some((Token::Colon, _)) = parser.pop_next() else {
            return Err(ParseError::with_reason(
                ParseErrorKind::MalformedVariable,
                span,
                "Expected colon",
            ));
        };
        let (ty, ty_span) = Type::parse(parser)?;
        let span = span.start..ty_span.end;
        Ok((Var { name, ty }, span))
    }
}

pub enum Statement {
    Let {
        var: Var,
        value: Option<Spanned<Expression>>,
    },
}

macro_rules! match_optional_token {
    ($parser:expr, $token_pattern:pat, $parse_fn:expr) => {
        match $parser.pop_next() {
            Some(($token_pattern, _)) => Some($parse_fn),
            Some(t) => {
                $parser.push_front(t);
                None
            }
            None => None,
        }
    };
}

impl Parseable for Statement {
    fn parse(parser: &mut Parser) -> Result<Spanned<Self>, ParseError> {
        if let Some((Token::Keyword(Keyword::Let), _)) = parser.peek_next() {
            parser.pop_next();
        } else {
            return Err(ParseError::spanless(ParseErrorKind::MalformedLet));
        }

        let (var, var_span) = Var::parse(parser)?;
        let value = match_optional_token!(parser, Token::Eq, Expression::parse(parser)?);
        let span = if let Some((_, span)) = &value {
            var_span.start..span.end
        } else {
            var_span
        };

        let Some((Token::SemiColon, _)) = parser.pop_next() else {
            return Err(ParseError::new(ParseErrorKind::ExpectedSemiColon, span));
        };

        Ok((Statement::Let { var, value }, span))
    }
}

pub enum Expression {
    Int(i32),
    String(String),
}

impl Parseable for Expression {
    fn parse(parser: &mut Parser) -> Result<Spanned<Self>, ParseError> {
        match parser.pop_next() {
            Some((Token::Int(i), span)) => Ok((Expression::Int(i), span)),
            // Some((Token::String(s), span)) => Ok((Expression::String(s), span)),
            Some(t) => {
                let span = t.1;
                Err(ParseError::with_reason(
                    ParseErrorKind::MalformedExpression,
                    span,
                    "Expected expression",
                ))
            }
            None => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }
}

pub enum Declaration {
    Function {
        name: String,
        arguments: Vec<Var>,
        body: Statement,
        ret_type: Option<Type>,
    },
}

pub struct Parser {
    tokens: VecDeque<Spanned<Token>>,
}

impl Parser {
    pub fn new(tokens: VecDeque<Spanned<Token>>) -> Self {
        Self { tokens }
    }

    fn expect_next(&mut self, token: Token) -> Result<Spanned<Token>, ParseError> {
        match self.tokens.pop_front() {
            Some(t) if t.0 == token => Ok(t),
            Some(t) => Err(ParseError::with_reason(
                ParseErrorKind::MalformedExpression,
                t.1,
                &format!("Expected `{:?}`, got `{:?}`", token, t.0),
            )),
            None => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }

    fn expect_kw(&mut self) -> Result<Spanned<Keyword>, ParseError> {
        match self.tokens.pop_front() {
            Some((Token::Keyword(kw), span)) => Ok((kw, span)),
            Some(t) => Err(ParseError::with_reason(
                ParseErrorKind::MalformedExpression,
                t.1,
                &format!("Expected keyword, got `{:?}`", t.0),
            )),
            None => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }

    fn pop_next(&mut self) -> Option<Spanned<Token>> {
        self.tokens.pop_front()
    }

    fn push_front(&mut self, token: Spanned<Token>) {
        self.tokens.push_front(token);
    }

    fn peek_next(&self) -> Option<Spanned<Token>> {
        self.tokens.front().cloned()
    }
}

impl Iterator for Parser {
    type Item = Spanned<Statement>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.tokens.is_empty() {
            None
        } else {
            match Statement::parse(self) {
                Ok(statement) => Some(statement),
                Err(err) => panic!("Parser error: {:?}", err),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser() {
        let src = "let a: int = 1;";
        let lexer = crate::parse::lex::tokenize(src);
        let toks = lexer.collect();
        println!("Toks: {:?}", toks);
        let parser = Parser::new(toks);

        let statements: Vec<Spanned<Statement>> = parser.into_iter().collect();
        assert_eq!(statements.len(), 1);
    }
}
