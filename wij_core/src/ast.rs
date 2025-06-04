mod macros;
pub mod typed;
mod types;
pub mod use_analysis;

use crate::Span;

use crate::{
    WijError,
    lex::{Keyword, Token},
    match_optional_token,
};

use typed::FunctionSignature;
pub use types::Type;

use std::{collections::VecDeque, fmt::Display};

pub type Spanned<T> = (T, Span);

#[allow(dead_code)]
pub fn span<T>(spanned: Spanned<T>) -> Span {
    spanned.1.clone()
}

#[allow(dead_code)]
pub fn t<T>(spanned: Spanned<T>) -> T {
    spanned.0
}

#[derive(Debug)]
pub enum ParseErrorKind {
    MalformedVariable,
    MalformedExpression,
    MalformedType,
    MalformedLet,
    ExpectedSemiColon,
    EndOfInput,
    TypeNameCapitalized,
    ComptimeArgOrder,
    InvalidVisibility,
}

impl Display for ParseErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
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

impl WijError for ParseError {
    fn span(&self) -> Option<Span> {
        self.span.clone()
    }

    fn reason(&self) -> String {
        match &self.reason {
            Some(reason) => reason.to_string(),
            None => self.kind.to_string(),
        }
    }

    fn notes(&self) -> Vec<(String, self::Span)> {
        Vec::new()
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
    pub ty: Option<Type>,
    pub is_comptime: bool,
}

impl Parseable for Var {
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        let mut is_comptime = false;
        let mut start_span = None;
        if let Some((Token::Keyword(Keyword::Comptime), comptime_span)) = parser.peek_next() {
            is_comptime = true;
            start_span = Some(comptime_span);
            parser.pop_next();
        }

        let (name, span) = parser.expect_ident()?;
        let span = match start_span {
            Some(start_span) => start_span.start..span.end,
            None => span,
        };

        let (ty, ty_span_end) = if let Some((Token::Colon, _)) = parser.peek_next() {
            parser.pop_next();
            let (ty, ty_span) = Type::parse(parser)?;
            (Some(ty), ty_span.end)
        } else {
            (None, span.end)
        };
        if is_comptime {
            if let Some(ty) = &ty {
                if ty != &Type::TypeType {
                    return Err(ParseError::with_reason(
                        ParseErrorKind::MalformedVariable,
                        span.start..ty_span_end,
                        "Comptime variables must be of type `type`",
                    ));
                }
            } else {
                return Err(ParseError::with_reason(
                    ParseErrorKind::MalformedVariable,
                    span.start..ty_span_end,
                    "Comptime variables must be of type `type`",
                ));
            }
        }

        let span = span.start..ty_span_end;
        Ok((
            Var {
                name,
                ty,
                is_comptime,
            },
            span,
        ))
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Literal {
    Int(i32),
    Usize(usize),
    Str(String),
    Bool(bool),
}

#[allow(unused)]
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Pat {
    Literal(Literal),
    TypeConstructor(String),
    Any(Option<String>), // corresponds to assigning any value to the variable, or _ if none
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MatchCase {
    pub pat: Pat,
    pub body: Spanned<Statement>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Statement {
    Let {
        var: Spanned<Var>,
        value: Option<Spanned<Expression>>,
    },
    Return(Option<Spanned<Expression>>),
    Block(Vec<Spanned<Statement>>),
    If {
        condition: Spanned<Expression>,
        then_block: Box<Spanned<Statement>>,
        else_block: Option<Box<Spanned<Statement>>>,
    },
    #[allow(unused)]
    Match {
        value: Spanned<Expression>,
        cases: Vec<Spanned<MatchCase>>,
    },
    For {
        var: Spanned<Var>,
        in_expr: Spanned<Expression>,
        body: Box<Spanned<Statement>>,
    },
    While {
        condition: Spanned<Expression>,
        body: Box<Spanned<Statement>>,
    },
    Break,
    Continue,
    Assignment(Box<Spanned<Expression>>, Box<Spanned<Expression>>),
    Expression(Box<Spanned<Expression>>), // a subclass of expressions being executed for side effects
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
                    var_span.clone()
                };

                let Some((Token::SemiColon, _)) = parser.pop_next() else {
                    return Err(ParseError::new(ParseErrorKind::ExpectedSemiColon, span));
                };

                Ok((
                    Statement::Let {
                        var: (var, var_span),
                        value,
                    },
                    span,
                ))
            }
            Some((Token::Keyword(Keyword::Break), span)) => {
                parser.pop_next();
                let _ = parser.expect_next(Token::SemiColon)?;
                Ok((Statement::Break, span))
            }
            Some((Token::Keyword(Keyword::Continue), span)) => {
                parser.pop_next();
                let _ = parser.expect_next(Token::SemiColon)?;
                Ok((Statement::Continue, span))
            }
            Some((Token::Keyword(Keyword::Return), span_start)) => {
                parser.pop_next();
                if let Some((Token::SemiColon, _)) = parser.peek_next() {
                    let _ = parser.expect_next(Token::SemiColon)?;
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
            Some((Token::Keyword(Keyword::For), for_start)) => {
                parser.pop_next();
                let var = Var::parse(parser)?;
                let _in = parser.expect_kw_kind(Keyword::In)?;
                let in_expr = Expression::parse(parser)?;
                let body = if let Some((Token::LBrace, _)) = parser.peek_next() {
                    Box::new(Statement::parse(parser)?)
                } else {
                    return Err(ParseError::with_reason(
                        ParseErrorKind::MalformedExpression,
                        for_start,
                        "Expected for block",
                    ));
                };
                let span = for_start.start..body.1.end;
                Ok((Statement::For { var, in_expr, body }, span))
            }
            Some((Token::Keyword(Keyword::While), while_start)) => {
                parser.pop_next();
                let condition = Expression::parse(parser)?;
                let body = if let Some((Token::LBrace, _)) = parser.peek_next() {
                    Box::new(Statement::parse(parser)?)
                } else {
                    return Err(ParseError::with_reason(
                        ParseErrorKind::MalformedExpression,
                        while_start,
                        "Expected while block",
                    ));
                };
                let span = while_start.start..body.1.end;
                Ok((Statement::While { condition, body }, span))
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
                if let Some((Token::Eq, _)) = parser.peek_next() {
                    parser.pop_next();
                    let rhs = Expression::parse(parser)?;
                    let expr_span = expr.1.clone();
                    let _ = parser.expect_next(Token::SemiColon)?;
                    let new_span = expr_span.start..rhs.1.end;
                    Ok((
                        Statement::Assignment(Box::new(expr), Box::new(rhs)),
                        new_span,
                    ))
                } else if expr.0.is_fn_call() {
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
    Mod,

    And,
    Or,
    EqEq,
    NEq,
    Gt,
    GtEq,
    Lt,
    LtEq,
}

impl Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::And => write!(f, "and"),
            BinOp::Or => write!(f, "or"),
            BinOp::EqEq => write!(f, "=="),
            BinOp::NEq => write!(f, "!="),
            BinOp::Gt => write!(f, ">"),
            BinOp::GtEq => write!(f, ">="),
            BinOp::Lt => write!(f, "<"),
            BinOp::LtEq => write!(f, "<="),
        }
    }
}

impl BinOp {
    fn precedence(&self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::EqEq | BinOp::NEq => 3,
            BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq => 4,
            BinOp::Add | BinOp::Sub => 5,
            BinOp::Mul | BinOp::Div => 6,
            BinOp::Mod => 7,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Expression {
    // Int(i32),
    // Bool(bool),
    // String(String),
    Literal(Literal),
    Array(Vec<Spanned<Expression>>),
    Ident(String),
    FieldAccess(Box<Spanned<Expression>>, String),
    MethodCall(Box<Spanned<Expression>>, String, Vec<Spanned<Expression>>),
    BinOp(BinOp, Box<Spanned<Expression>>, Box<Spanned<Expression>>),
    FnCall(String, Vec<Spanned<Expression>>),
    RecordInit(String, Vec<(String, Spanned<Expression>)>),
    DataConstruction(String, Option<Box<Spanned<Expression>>>),
    Idx(Box<Spanned<Expression>>, Box<Spanned<Expression>>),
    // comptime intrinsics are denoted by @
    // they can have optional arguments
    Type(Type), // these great erased prior to SSA conversion
    Intrinsic(String, Vec<Spanned<Expression>>),
    Self_,
}

fn parse_record_assignments(
    parser: &mut Parser,
) -> ParseResult<Vec<(String, Spanned<Expression>)>> {
    let mut assignments = Vec::new();
    loop {
        let ident = parser.expect_ident()?.0;
        let _ = parser.expect_next(Token::Eq)?;
        let expr = Expression::parse(parser)?;
        assignments.push((ident, expr));
        if let Some((Token::Comma, _)) = parser.peek_next() {
            parser.pop_next();
        } else {
            break;
        }
    }
    Ok(assignments)
}

fn parse_array_elems(parser: &mut Parser) -> ParseResult<Vec<Spanned<Expression>>> {
    let mut elems = Vec::new();
    loop {
        let expr = Expression::parse(parser)?;
        elems.push(expr);
        if let Some((Token::Comma, _)) = parser.peek_next() {
            parser.pop_next();
        } else {
            break;
        }
    }
    Ok(elems)
}

fn parse_fn_call_args(parser: &mut Parser) -> ParseResult<Vec<Spanned<Expression>>> {
    let mut args = vec![];
    // todo: clean this up a bit
    if let Some((Token::RParen, _)) = parser.peek_next() {
        return Ok(args);
    }

    loop {
        let arg = Expression::parse(parser)?;
        let arg_span = arg.1.clone();
        args.push(arg);

        match parser.peek_next() {
            Some((Token::Comma, _)) => {
                parser.pop_next();
            }
            Some((Token::RParen, _)) => {
                break;
            }
            _ => {
                return Err(ParseError::with_reason(
                    ParseErrorKind::MalformedExpression,
                    arg_span,
                    "Expected ',' or ')' in function arguments",
                ));
            }
        }
    }

    Ok(args)
}

impl Expression {
    fn is_fn_call(&self) -> bool {
        matches!(self, Expression::FnCall(_, _))
    }

    fn parse_primary(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        use Literal::*;

        let mut expr = match parser.pop_next() {
            Some((Token::Keyword(Keyword::Self_), span)) => (Expression::Self_, span),
            Some((Token::Int(i), span)) => (Expression::Literal(Int(i)), span),
            Some((Token::Keyword(Keyword::True), span)) => (Expression::Literal(Bool(true)), span),
            Some((Token::Keyword(Keyword::False), span)) => {
                (Expression::Literal(Bool(false)), span)
            }
            Some((Token::String(s), span)) => (Expression::Literal(Str(s)), span),
            Some((Token::LBracket, start_span)) => {
                let elems = parse_array_elems(parser)?;
                let end_span = parser.expect_next(Token::RBracket)?;
                (Expression::Array(elems), start_span.start..end_span.1.end)
            }
            Some((Token::At, span)) => {
                let ident = parser.expect_ident()?.0;
                let _lparen = parser.expect_next(Token::LParen)?;
                let args = parse_fn_call_args(parser)?;
                let (_, rparen_span) = parser.expect_next(Token::RParen)?;
                (
                    Expression::Intrinsic(ident, args),
                    span.start..rparen_span.end,
                )
            }
            Some((Token::Identifier(ident), span)) => match parser.peek_next() {
                Some((Token::LParen, _)) => {
                    parser.pop_next();

                    if let Some((Token::RParen, end_span)) = parser.peek_next() {
                        parser.pop_next();
                        let full_span = span.start..end_span.end;
                        (Expression::FnCall(ident, vec![]), full_span)
                    } else {
                        let args = parse_fn_call_args(parser)?;
                        let (_, rparen_span) = parser.expect_next(Token::RParen)?;
                        let full_span = span.start..rparen_span.end;
                        (Expression::FnCall(ident, args), full_span)
                    }
                }
                Some((Token::LBrace, _)) => {
                    if ident.chars().next().unwrap().is_uppercase() {
                        parser.pop_next();
                        let assignments = parse_record_assignments(parser)?;
                        let (_, rbrace_span) = parser.expect_next(Token::RBrace)?;
                        let full_span = span.start..rbrace_span.end;
                        (Expression::RecordInit(ident, assignments), full_span)
                    } else {
                        (Expression::Ident(ident), span)
                    }
                }
                _ => {
                    if ident.chars().next().unwrap().is_uppercase() {
                        let mut try_parser = parser.clone();
                        match Expression::parse(&mut try_parser) {
                            Ok((e, e_span)) => {
                                parser.replace(try_parser);
                                (
                                    Expression::DataConstruction(
                                        ident,
                                        Some(Box::new((e, e_span.clone()))),
                                    ),
                                    span.start..e_span.end,
                                )
                            }
                            Err(_) => (Expression::Type(Type::UserDef(ident)), span),
                        }
                    } else {
                        (Expression::Ident(ident), span)
                    }
                }
            },
            Some((Token::LParen, lparen_span)) => {
                // handle parenthesized expressions
                let expr = Expression::parse(parser)?;
                let (_, rparen_span) = parser.expect_next(Token::RParen)?;
                let full_span = lparen_span.start..rparen_span.end;
                (expr.0, full_span) // return the inner expression with updated span
            }
            Some((token, span)) => {
                parser.push_front((token.clone(), span.clone()));

                match Type::parse(parser) {
                    Ok((ty, span)) => (Expression::Type(ty), span),
                    Err(_e) => {
                        return Err(ParseError::with_reason(
                            ParseErrorKind::MalformedExpression,
                            span,
                            &format!("Expected expression, got {:?}", token),
                        ));
                    }
                }
            }
            None => return Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        };

        while let Some((Token::Dot, _)) = parser.peek_next() {
            parser.pop_next();

            if let Some((Token::Identifier(field_name), field_span)) = parser.pop_next() {
                if let Some((Token::LParen, _)) = parser.peek_next() {
                    parser.pop_next();
                    let args = parse_fn_call_args(parser)?;
                    let (_, rparen) = parser.expect_next(Token::RParen)?;
                    let span = expr.1.start..rparen.end;
                    expr = (
                        Expression::MethodCall(Box::new(expr), field_name, args),
                        span,
                    );
                    break;
                }

                let start = expr.1.start;
                let end = field_span.end;
                expr = (
                    Expression::FieldAccess(Box::new(expr), field_name),
                    start..end,
                );
            } else {
                return Err(ParseError::with_reason(
                    ParseErrorKind::MalformedExpression,
                    expr.1.clone(),
                    "Expected field name after '.'",
                ));
            }
        }

        if let Some((Token::LBracket, lbracket_span)) = parser.peek_next() {
            parser.pop_next();
            let index = Expression::parse(parser)?;
            let (_, rbracket_span) = parser.expect_next(Token::RBracket)?;
            let full_span = lbracket_span.start..rbracket_span.end;
            expr = (Expression::Idx(Box::new(expr), Box::new(index)), full_span);
        }

        Ok(expr)
    }

    // parse bin ops using precedence climbing
    fn parse_binary_op_rhs(
        parser: &mut Parser,
        mut lhs: Spanned<Self>,
        min_precedence: u8,
    ) -> ParseResult<Spanned<Self>> {
        // look ahead for binops
        while let Some((Token::BinOp(op), op_span)) = parser.peek_next() {
            let op_precedence = op.precedence();

            // only handle this operator if it has high enough precedence
            if op_precedence < min_precedence {
                break;
            }

            if parser.pop_next().is_none() {
                return Err(ParseError::new(ParseErrorKind::EndOfInput, op_span));
            }

            let mut rhs = Self::parse_primary(parser)?;

            while let Some((Token::BinOp(next_op), _)) = parser.peek_next() {
                let next_precedence = next_op.precedence();

                // if the next operator has higher precedence, we need to recursively parse it first
                if next_precedence > op_precedence {
                    rhs = Self::parse_binary_op_rhs(parser, rhs, next_precedence)?;
                    continue;
                }
                break;
            }

            let full_span = lhs.1.start..rhs.1.end;
            lhs = (
                Expression::BinOp(op, Box::new(lhs), Box::new(rhs)),
                full_span,
            );
            continue;
        }

        Ok(lhs)
    }
}

impl Parseable for Expression {
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        let lhs = Self::parse_primary(parser)?;
        Self::parse_binary_op_rhs(parser, lhs, 0)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ForeignDeclaration {
    pub name: String,
    pub sig: FunctionSignature,
}

impl Parseable for ForeignDeclaration {
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        let (name, span) = parser.expect_ident()?;

        let _colon = parser.expect_next(Token::Colon)?;
        let _fn_kw = parser.expect_kw_kind(Keyword::Fn)?;
        let _lparen = parser.expect_next(Token::LParen)?;
        let args = parse_args(parser, Token::RParen)?;
        let ret_type = if let Some((Token::Arrow, _)) = parser.peek_next() {
            parser.pop_next();
            Some(Type::parse(parser)?.0)
        } else {
            None
        };
        let _semi = parser.expect_next(Token::SemiColon)?;

        let param_types = args
            .iter()
            .map(|(arg, _)| arg.ty.clone().unwrap())
            .collect();
        let ret_type = match ret_type {
            Some(t) => t,
            None => Type::Unit,
        };

        let func_sig = FunctionSignature {
            param_types,
            ret_type,
        };

        Ok((
            ForeignDeclaration {
                name,
                sig: func_sig,
            },
            span,
        ))
    }
}

pub type Path = Vec<String>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct EnumVariant {
    pub name: String,
    pub data: Option<Spanned<Type>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Function {
    name: String,
    arguments: Vec<Spanned<Var>>,
    num_comptime_args: usize,
    body: Spanned<Statement>,
    ret_type: Option<Type>,
}

impl Parseable for Function {
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        let start_span = parser.expect_kw_kind(Keyword::Fn)?;
        let (name, _) = parser.expect_ident()?;
        let _lparen = parser.expect_next(Token::LParen)?;
        let arguments = parse_args(parser, Token::RParen)?;
        let mut num_comptime_args = 0;
        // could probably do this better
        let mut still_looking_for_comptime = true;
        for (arg, span) in &arguments {
            if arg.is_comptime {
                if !still_looking_for_comptime {
                    return Err(ParseError::with_reason(
                        ParseErrorKind::ComptimeArgOrder,
                        span.clone(),
                        "Comptime arguments must come first",
                    ));
                }
                num_comptime_args += 1;
            } else {
                still_looking_for_comptime = false;
            }
        }
        let ret_type = if let Some((Token::Arrow, _)) = parser.peek_next() {
            parser.pop_next();
            Some(Type::parse(parser)?.0)
        } else {
            None
        };
        let (body, body_span) = Statement::parse(parser)?;
        let span = start_span.start..body_span.end;
        Ok((
            Function {
                name,
                arguments,
                num_comptime_args,
                body: (body, body_span),
                ret_type,
            },
            span,
        ))
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum DeclKind {
    Function(Function),
    Record {
        name: String,
        fields: Vec<Spanned<Var>>,
    },
    Enum {
        name: String,
        variants: Vec<Spanned<EnumVariant>>,
    },
    Module(String),
    ForeignDeclarations(Vec<Spanned<ForeignDeclaration>>),
    Use(Path),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Visibility {
    Public,
    Private,
}

impl Visibility {
    pub(crate) fn to_bool(&self) -> bool {
        match self {
            Visibility::Public => true,
            Visibility::Private => false,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Declaration {
    pub visibility: Visibility,
    pub decl: DeclKind,
}

impl Declaration {
    fn name(&self) -> Option<String> {
        match &self.decl {
            DeclKind::Function(Function { name, .. }) => Some(name.clone()),
            DeclKind::Record { name, .. } => Some(name.clone()),
            DeclKind::Enum { name, .. } => Some(name.clone()),
            _ => None,
        }
    }
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
        let var = Var::parse(parser)?;
        if var.0.ty.is_none() {
            return Err(ParseError::with_reason(
                ParseErrorKind::MalformedType,
                var.1.clone(),
                "Expected type annotation",
            ));
        }
        vars.push(var);
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
            Some((Token::Keyword(Keyword::Module), _)) => {
                let mod_span = parser.expect_kw_kind(Keyword::Module)?;
                let (name, name_span) = parser.expect_ident()?;
                let _ = parser.expect_next(Token::SemiColon)?;
                let span = mod_span.start..name_span.end;
                let decl = Declaration {
                    visibility: Visibility::Private,
                    decl: DeclKind::Module(name),
                };
                Ok((decl, span))
            }
            Some((Token::Keyword(Keyword::Foreign), _)) => {
                let _foreign_span = parser.expect_kw_kind(Keyword::Foreign)?;
                let _lbrace = parser.expect_next(Token::LBrace)?;

                let mut foreign_decls = vec![];
                while parser.peek_next().is_some() {
                    if let Some((Token::RBrace, _)) = parser.peek_next() {
                        break;
                    }
                    foreign_decls.push(ForeignDeclaration::parse(parser)?);
                }

                let _rbrace = parser.expect_next(Token::RBrace)?;
                let span = _foreign_span.start.._rbrace.1.end;

                let decl = Declaration {
                    visibility: Visibility::Private,
                    decl: DeclKind::ForeignDeclarations(foreign_decls),
                };
                Ok((decl, span))
            }
            Some((Token::Keyword(Keyword::Use), use_span)) => {
                let _use_span = parser.expect_kw_kind(Keyword::Use)?;
                let (name, _) = parser.expect_ident()?;
                let mut path = vec![name];
                while let Some((Token::Colon, _)) = parser.peek_next() {
                    parser.pop_next();
                    let (name, _) = parser.expect_ident()?;
                    path.push(name);
                }
                let (_, semi_span) = parser.expect_next(Token::SemiColon)?;
                let span = use_span.start..semi_span.end;

                let decl = Declaration {
                    visibility: Visibility::Private,
                    decl: DeclKind::Use(path),
                };
                Ok((decl, span))
            }
            Some((Token::Keyword(Keyword::Fn), _)) => {
                let (func, func_span) = Function::parse(parser)?;
                let decl = Declaration {
                    visibility: Visibility::Private,
                    decl: DeclKind::Function(func),
                };
                Ok((decl, func_span))
            }
            Some((Token::Keyword(Keyword::Pub), _)) => {
                let pub_span = parser.expect_kw_kind(Keyword::Pub)?;
                let (mut decl, decl_span) = Declaration::parse(parser)?;
                match decl.decl {
                    DeclKind::Function { .. } | DeclKind::Record { .. } | DeclKind::Enum { .. } => {
                    }
                    _ => {
                        return Err(ParseError::new(
                            ParseErrorKind::InvalidVisibility,
                            decl_span,
                        ));
                    }
                }
                decl.visibility = Visibility::Public;

                Ok((decl, pub_span))
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
                        let decl = Declaration {
                            visibility: Visibility::Private,
                            decl: DeclKind::Record {
                                name: type_name,
                                fields,
                            },
                        };
                        Ok((decl, span))
                    }
                    Some(_) => {
                        // Enum (probably)
                        let mut variants = Vec::new();
                        loop {
                            let (variant_name, variant_span) = parser.expect_ident()?;
                            let mut variant = EnumVariant {
                                name: variant_name,
                                data: None,
                            };
                            match parser.peek_next() {
                                Some((Token::Bar, _)) => {
                                    variants.push((variant, variant_span));
                                    parser.pop_next();
                                }
                                Some((Token::SemiColon, _)) => {
                                    variants.push((variant, variant_span));
                                    parser.pop_next();
                                    break;
                                }
                                Some((t, ty_span)) => {
                                    let ty = match Type::parse(parser) {
                                        Ok(t) => t,
                                        Err(_) => {
                                            return Err(ParseError::with_reason(
                                                ParseErrorKind::MalformedExpression,
                                                ty_span,
                                                &format!(
                                                    "Expected semicolon, bar or type, got {:?}",
                                                    t
                                                ),
                                            ));
                                        }
                                    };

                                    variant.data = Some(ty);
                                    variants.push((variant, variant_span.start..ty_span.end));

                                    match parser.peek_next() {
                                        Some((Token::Bar, _)) => {
                                            parser.pop_next();
                                        }
                                        Some((Token::SemiColon, _)) => {
                                            parser.pop_next();
                                            break;
                                        }
                                        tt => {
                                            return Err(ParseError::with_reason(
                                                ParseErrorKind::MalformedExpression,
                                                ty_span,
                                                &format!(
                                                    "Expected semicolon, bar or type, got {:?}",
                                                    tt
                                                ),
                                            ));
                                        }
                                    }
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
                        let decl = Declaration {
                            visibility: Visibility::Private,
                            decl: DeclKind::Enum {
                                name: type_name,
                                variants,
                            },
                        };
                        Ok((decl, type_span))
                    }
                    None => Err(ParseError::new(ParseErrorKind::EndOfInput, ty_span_start)),
                }
            }
            _ => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Parser {
    tokens: VecDeque<Spanned<Token>>,
}

impl Parser {
    pub fn new(tokens: VecDeque<Spanned<Token>>) -> Self {
        Self { tokens }
    }

    fn replace(&mut self, other: Parser) {
        self.tokens = other.tokens;
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

    #[allow(dead_code)]
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
mod tests;
