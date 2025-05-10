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
    TypeNameCapitalized,
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
pub type ParseResult<T> = Result<T, ParseError>;

pub trait Parseable
where
    Self: Sized,
{
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>>;
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Var {
    pub name: String,
    pub ty: Type,
}

impl Parseable for Var {
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        let (name, span) = parser.expect_ident()?;

        let Some((Token::Colon, _)) = parser.pop_next() else {
            return Err(ParseError::with_reason(
                ParseErrorKind::MalformedVariable,
                span,
                "Expected type",
            ));
        };
        let (ty, ty_span) = Type::parse(parser)?;
        let span = span.start..ty_span.end;
        Ok((Var { name, ty }, span))
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Statement {
    Let {
        var: Var,
        value: Option<Spanned<Expression>>,
    },
    Return(Option<Spanned<Expression>>),
    Block(Vec<Spanned<Statement>>),
    If {
        condition: Spanned<Expression>,
        then_block: Box<Spanned<Statement>>,
        else_block: Option<Box<Spanned<Statement>>>,
    },
    Expression(Box<Spanned<Expression>>), // a subclass of expressions being executed for side effects
}

impl Statement {
    fn is_block(&self) -> bool {
        matches!(self, Statement::Block(_))
    }
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
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        match parser.peek_next() {
            Some((Token::Keyword(Keyword::Let), _)) => {
                parser.pop_next();
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
            Some((Token::Keyword(Keyword::Return), span_start)) => {
                parser.pop_next();
                if let Some((Token::SemiColon, _)) = parser.peek_next() {
                    Ok((
                        Statement::Return(None),
                        span_start.start..span_start.end + 1,
                    ))
                } else {
                    let (expr, expr_span) = Expression::parse(parser)?;
                    let _ = parser.expect_next(Token::SemiColon)?;
                    let expr_end = expr_span.end;
                    Ok((
                        Statement::Return(Some((expr, expr_span))),
                        span_start.start..expr_end,
                    ))
                }
            }
            Some((Token::Keyword(Keyword::If), if_start)) => {
                parser.pop_next();
                let cond = Expression::parse(parser)?;
                let then_block = if let Some((Token::LBrace, _)) = parser.peek_next() {
                    Box::new(Statement::parse(parser)?)
                } else {
                    return Err(ParseError::with_reason(
                        ParseErrorKind::MalformedExpression,
                        if_start,
                        "Expected then block",
                    ));
                };

                if let Some((Token::Keyword(Keyword::Else), _)) = parser.peek_next() {
                    parser.pop_next();
                    let else_block = if let Some((Token::LBrace, _)) = parser.peek_next() {
                        Some(Box::new(Statement::parse(parser)?))
                    } else {
                        return Err(ParseError::with_reason(
                            ParseErrorKind::MalformedExpression,
                            if_start,
                            "Expected else block",
                        ));
                    };
                    let if_span = if_start.start..else_block.as_ref().unwrap().1.end;
                    Ok((
                        Statement::If {
                            condition: cond,
                            then_block,
                            else_block,
                        },
                        if_span,
                    ))
                } else {
                    let if_span = if_start.start..then_block.1.end;
                    Ok((
                        Statement::If {
                            condition: cond,
                            then_block,
                            else_block: None,
                        },
                        if_span,
                    ))
                }
            }
            Some((Token::LBrace, span)) => {
                parser.pop_next();

                let mut stmts = Vec::new();
                let mut last_span = span.clone();
                while parser.has_next() {
                    if let Some((Token::RBrace, _)) = parser.peek_next() {
                        parser.pop_next();
                        break;
                    }
                    let stmt = Statement::parse(parser)?;
                    last_span = stmt.1.clone();
                    stmts.push(stmt);
                }

                let block_span = span.start..last_span.end;
                Ok((Statement::Block(stmts), block_span))
            }
            Some(t) => {
                let expr = Expression::parse(parser)?;
                if expr.0.is_fn_call() {
                    let _ = parser.expect_next(Token::SemiColon)?;
                    let expr_span = expr.1.clone();
                    Ok((Statement::Expression(Box::new(expr)), expr_span))
                } else {
                    let span = t.1;
                    Err(ParseError::with_reason(
                        ParseErrorKind::MalformedExpression,
                        span,
                        &format!("Expected statement, got {:?}", t.0),
                    ))
                }
            }

            None => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,

    And,
    Or,
    EqEq,
    NEq,
    Gt,
    GtEq,
    Lt,
    LtEq,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Expression {
    Int(i32),
    String(String),
    Ident(String),
    BinOp(BinOp, Box<Spanned<Expression>>, Box<Spanned<Expression>>),
    FnCall(String, Vec<Spanned<Expression>>),
}

impl Expression {
    fn is_fn_call(&self) -> bool {
        matches!(self, Expression::FnCall(_, _))
    }
}

impl Parseable for Expression {
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        match parser.pop_next() {
            Some((Token::Int(i), span)) => Ok((Expression::Int(i), span)),
            // Some((Token::String(s), span)) => Ok((Expression::String(s), span)),
            Some((Token::Identifier(ident), span)) => match parser.peek_next() {
                Some((Token::LParen, _)) => {
                    parser.pop_next();
                    let mut args = Vec::new();
                    loop {
                        let arg = Expression::parse(parser)?;
                        args.push(arg);
                        if let Some((Token::Comma, _)) = parser.peek_next() {
                            parser.pop_next();
                        } else if let Some((Token::RParen, _)) = parser.peek_next() {
                            break;
                        }
                    }

                    let rparen = parser.expect_next(Token::RParen)?;
                    let span = span.start..rparen.1.end;
                    Ok((Expression::FnCall(ident, args), span))
                }
                _ => Ok((Expression::Ident(ident), span)),
            },
            Some(t) => {
                let span = t.1;
                Err(ParseError::with_reason(
                    ParseErrorKind::MalformedExpression,
                    span,
                    &format!("Expected expression, got {:?}", t.0),
                ))
            }
            None => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Declaration {
    Function {
        name: String,
        arguments: Vec<Spanned<Var>>,
        body: Statement,
        ret_type: Option<Type>,
    },
    Record {
        name: String,
        fields: Vec<Spanned<Var>>,
    },
    Enum {
        name: String,
        variants: Vec<Spanned<String>>,
    },
}

fn parse_args(parser: &mut Parser, last_token: Token) -> ParseResult<Vec<Spanned<Var>>> {
    let mut vars = Vec::new();
    while parser.peek_next().is_some() {
        if let Some((lt, _)) = parser.peek_next() {
            if lt == last_token {
                parser.pop_next();
                break;
            }
        }
        vars.push(Var::parse(parser)?);
        if let Some((Token::Comma, _)) = parser.peek_next() {
            parser.pop_next();
        } else if let Some((lt, _)) = parser.peek_next() {
            if lt == last_token {
                parser.pop_next();
                break;
            }
        }
    }

    Ok(vars)
}

impl Parseable for Declaration {
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        match parser.peek_next() {
            Some((Token::Keyword(Keyword::Fn), _)) => {
                let start_span = parser.expect_kw_kind(Keyword::Fn)?;
                let (name, _) = parser.expect_ident()?;
                let _lparen = parser.expect_next(Token::LParen)?;
                let arguments = parse_args(parser, Token::RParen)?;
                let ret_type = if let Some((Token::Arrow, _)) = parser.peek_next() {
                    Some(Type::parse(parser)?.0)
                } else {
                    None
                };
                let (body, body_span) = Statement::parse(parser)?;
                let span = start_span.start..body_span.end;
                Ok((
                    Declaration::Function {
                        name,
                        arguments,
                        body,
                        ret_type,
                    },
                    span,
                ))
            }
            Some((Token::Keyword(Keyword::Type), ty_span_start)) => {
                let _start_span = parser.expect_kw_kind(Keyword::Type)?;
                let (type_name, type_span) = parser.expect_ident()?;
                if !type_name.chars().next().unwrap().is_uppercase() {
                    return Err(ParseError::with_reason(
                        ParseErrorKind::TypeNameCapitalized,
                        type_span,
                        "Type name must start with an uppercase letter",
                    ));
                }

                let _eq = parser.expect_next(Token::Eq)?;
                match parser.peek_next() {
                    Some((Token::LBrace, _)) => {
                        // Record
                        parser.pop_next();
                        let fields = parse_args(parser, Token::RBrace)?;
                        let span = ty_span_start.start..fields.last().unwrap().1.end;
                        Ok((
                            Declaration::Record {
                                name: type_name,
                                fields,
                            },
                            span,
                        ))
                    }
                    Some(_) => {
                        // Enum (probably)
                        let mut variants = Vec::new();
                        loop {
                            let (variant_name, variant_span) = parser.expect_ident()?;
                            match parser.peek_next() {
                                Some((Token::Bar, _)) => {
                                    variants.push((variant_name, variant_span));
                                    parser.pop_next();
                                }
                                Some((Token::SemiColon, _)) => {
                                    variants.push((variant_name, variant_span));
                                    parser.pop_next();
                                    break;
                                }
                                Some((Token::LBrace, _)) => {
                                    // Record
                                    unimplemented!()
                                    // let fields = parse_args(parser, Token::RBrace)?;
                                    // variants.push((variant_name, fields));
                                    // break;
                                }
                                Some((t, span)) => {
                                    return Err(ParseError::with_reason(
                                        ParseErrorKind::MalformedExpression,
                                        span,
                                        &format!("Expected semicolon or bar, got {:?}", t),
                                    ));
                                }
                                None => {
                                    return Err(ParseError::new(
                                        ParseErrorKind::EndOfInput,
                                        variant_span.end..variant_span.end,
                                    ));
                                }
                            }
                        }

                        let type_span = ty_span_start.start..variants.last().unwrap().1.end;
                        Ok((
                            Declaration::Enum {
                                name: type_name,
                                variants,
                            },
                            type_span,
                        ))
                    }
                    None => Err(ParseError::new(ParseErrorKind::EndOfInput, ty_span_start)),
                }
            }
            _ => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }
}

pub struct Parser {
    tokens: VecDeque<Spanned<Token>>,
}

impl Parser {
    pub fn new(tokens: VecDeque<Spanned<Token>>) -> Self {
        Self { tokens }
    }

    fn has_next(&self) -> bool {
        !self.tokens.is_empty()
    }

    fn expect_next(&mut self, token: Token) -> ParseResult<Spanned<Token>> {
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

    fn expect_kw(&mut self) -> ParseResult<Spanned<Keyword>> {
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

    fn expect_kw_kind(&mut self, kw: Keyword) -> ParseResult<Span> {
        match self.expect_kw() {
            Ok((kw2, span)) if kw == kw2 => Ok(span),
            Ok((kw2, span)) => Err(ParseError::with_reason(
                ParseErrorKind::MalformedExpression,
                span,
                &format!("Expected `{:?}`, got `{:?}`", kw, kw2),
            )),
            Err(err) => Err(err),
        }
    }

    fn expect_ident(&mut self) -> ParseResult<(String, Span)> {
        match self.tokens.pop_front() {
            Some((Token::Identifier(ident), span)) => Ok((ident, span)),
            Some(t) => Err(ParseError::with_reason(
                ParseErrorKind::MalformedExpression,
                t.1,
                &format!("Expected identifier, got `{:?}`", t.0),
            )),
            None => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }

    fn take_until(&mut self, until_tok: Token) -> ParseResult<Vec<Spanned<Token>>> {
        let mut tokens = Vec::new();
        while let Some(t) = self.pop_next() {
            if t.0 == until_tok {
                break;
            }
            tokens.push(t);
        }
        Ok(tokens)
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
    type Item = ParseResult<Spanned<Declaration>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.tokens.is_empty() {
            None
        } else {
            Some(Declaration::parse(self))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_parser() {
    //     let src = "let a: int = 1;";
    //     let lexer = crate::parse::lex::tokenize(src);
    //     let toks = lexer.collect();
    //     println!("Toks: {:?}", toks);
    //     let parser = Parser::new(toks);
    //
    //     let statements: Vec<Spanned<Statement>> = parser.into_iter().collect();
    //     assert_eq!(statements.len(), 1);
    // }

    #[test]
    fn test_parse_fn() {
        let src = "fn main() {}";
        let lexer = crate::parse::lex::tokenize(src);
        let toks = lexer.collect();
        let parser = Parser::new(toks);

        let decls = parser
            .map(|r| r.unwrap())
            .collect::<Vec<Spanned<Declaration>>>();
        let expected = vec![(
            Declaration::Function {
                name: "main".to_string(),
                arguments: vec![],
                body: Statement::Block(vec![]),
                ret_type: None,
            },
            0..11,
        )];
        assert_eq!(decls.len(), 1);
        assert_eq!(decls, expected);
    }

    #[test]
    fn test_parse_fn_with_params_and_body() {
        let src = "fn main(a: int, b: int) {
            let c: int = 1;
        }";
        let lexer = crate::parse::lex::tokenize(src);
        let toks = lexer.collect();
        let parser = Parser::new(toks);

        let decls = parser
            .map(|r| r.unwrap())
            .collect::<Vec<Spanned<Declaration>>>();
        let expected = vec![(
            Declaration::Function {
                name: "main".to_string(),
                arguments: vec![
                    (
                        Var {
                            name: "a".to_string(),
                            ty: Type::Int,
                        },
                        8..14,
                    ),
                    (
                        Var {
                            name: "b".to_string(),
                            ty: Type::Int,
                        },
                        16..22,
                    ),
                ],
                body: Statement::Block(vec![(
                    Statement::Let {
                        var: Var {
                            name: "c".to_string(),
                            ty: Type::Int,
                        },
                        value: Some((Expression::Int(1), 51..52)),
                    },
                    42..52,
                )]),
                ret_type: None,
            },
            0..52,
        )];
        assert_eq!(decls.len(), 1);
        assert_eq!(decls, expected);
    }

    #[test]
    fn test_return_fn() {
        let src = "fn main() {return 1;}";
        let lexer = crate::parse::lex::tokenize(src);
        let toks = lexer.collect();
        let parser = Parser::new(toks);

        let decls = parser
            .map(|r| r.unwrap())
            .collect::<Vec<Spanned<Declaration>>>();
        let expected = vec![(
            Declaration::Function {
                name: "main".to_string(),
                arguments: vec![],
                body: Statement::Block(vec![(
                    Statement::Return(Some((Expression::Int(1), 18..19))),
                    11..19,
                )]),
                ret_type: None,
            },
            0..19,
        )];
        assert_eq!(decls.len(), expected.len());
        assert_eq!(decls, expected);
    }

    #[test]
    fn test_if_else() {
        let src = "fn main() { if a { return 1; } else { return 2; }}";
        let lexer = crate::parse::lex::tokenize(src);
        let toks = lexer.collect();
        let parser = Parser::new(toks);

        let decls = parser
            .map(|r| r.unwrap())
            .collect::<Vec<Spanned<Declaration>>>();

        let expected = vec![(
            Declaration::Function {
                name: "main".to_string(),
                arguments: vec![],
                body: Statement::Block(vec![(
                    Statement::If {
                        condition: (Expression::Ident("a".to_string()), 15..16),
                        then_block: Box::new((
                            Statement::Block(vec![(
                                Statement::Return(Some((Expression::Int(1), 26..27))),
                                19..27,
                            )]),
                            17..27,
                        )),
                        else_block: Some(Box::new((
                            Statement::Block(vec![(
                                Statement::Return(Some((Expression::Int(2), 45..46))),
                                38..46,
                            )]),
                            36..46,
                        ))),
                    },
                    12..46,
                )]),
                ret_type: None,
            },
            0..46,
        )];

        assert_eq!(decls.len(), expected.len());
        assert_eq!(decls, expected);
    }

    #[test]
    fn test_type_enum() {
        let src = "type Test = Red;";
        let lexer = crate::parse::lex::tokenize(src);
        let toks = lexer.collect();
        let parser = Parser::new(toks);

        let decls = parser
            .map(|r| r.unwrap())
            .collect::<Vec<Spanned<Declaration>>>();
        let expected = vec![(
            Declaration::Enum {
                name: "Test".to_string(),
                variants: vec![("Red".to_string(), 12..15)],
            },
            0..15,
        )];
        assert_eq!(decls.len(), expected.len());
        assert_eq!(decls, expected);
    }

    #[test]
    fn test_type_record() {
        let src = "type Test = { a: int, b: int }";

        let lexer = crate::parse::lex::tokenize(src);
        let toks = lexer.collect();
        let parser = Parser::new(toks);

        let decls = parser
            .map(|r| r.unwrap())
            .collect::<Vec<Spanned<Declaration>>>();
        let expected = vec![(
            Declaration::Record {
                name: "Test".to_string(),
                fields: vec![
                    (
                        Var {
                            name: "a".to_string(),
                            ty: Type::Int,
                        },
                        14..20,
                    ),
                    (
                        Var {
                            name: "b".to_string(),
                            ty: Type::Int,
                        },
                        22..28,
                    ),
                ],
            },
            0..28,
        )];
        assert_eq!(decls.len(), expected.len());
        assert_eq!(decls, expected);
    }

    #[test]
    fn test_fn_call() {
        let src = "fn main() { let a: int = add(1, 2); }";
        let lexer = crate::parse::lex::tokenize(src);
        let toks = lexer.collect();
        let parser = Parser::new(toks);

        let decls = parser
            .map(|r| r.unwrap())
            .collect::<Vec<Spanned<Declaration>>>();
        let expected = vec![(
            Declaration::Function {
                name: "main".to_string(),
                arguments: vec![],
                body: Statement::Block(vec![(
                    Statement::Let {
                        var: Var {
                            name: "a".to_string(),
                            ty: Type::Int,
                        },
                        value: Some((
                            Expression::FnCall(
                                "add".to_string(),
                                vec![(Expression::Int(1), 29..30), (Expression::Int(2), 32..33)],
                            ),
                            25..34,
                        )),
                    },
                    16..34,
                )]),
                ret_type: None,
            },
            0..34,
        )];
        assert_eq!(decls.len(), expected.len());
        assert_eq!(decls, expected);
    }

    #[test]
    fn test_fn_call_as_stmt() {
        let src = "fn main() { add(1, 2); }";
        let lexer = crate::parse::lex::tokenize(src);
        let toks = lexer.collect();
        let parser = Parser::new(toks);

        let decls = parser
            .map(|r| r.unwrap())
            .collect::<Vec<Spanned<Declaration>>>();
        let expected = vec![(
            Declaration::Function {
                name: "main".to_string(),
                arguments: vec![],
                body: Statement::Block(vec![(
                    Statement::Expression(Box::new((
                        Expression::FnCall(
                            "add".to_string(),
                            vec![(Expression::Int(1), 16..17), (Expression::Int(2), 19..20)],
                        ),
                        12..21,
                    ))),
                    12..21,
                )]),
                ret_type: None,
            },
            0..21,
        )];
        assert_eq!(decls.len(), expected.len());
        assert_eq!(decls, expected);
    }
}
