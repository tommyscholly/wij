use std::{collections::HashMap, fmt::Display};

use crate::{
    SizeOf,
    lex::{Keyword, Token},
};

use super::{
    BinOp, ParseError, ParseErrorKind, Parseable, Parser, Spanned, typed::FunctionSignature,
};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    Int,
    Usize,
    Bool,
    Str,
    Byte,
    #[allow(clippy::enum_variant_names)]
    TypeType, // A type type is only for comptime operations
    OpaquePtr,
    Ptr(Box<Type>),
    Array(Box<Type>),
    Tuple(Vec<Type>),
    Fn(Box<FunctionSignature>),
    UserDef(String),
    Record(Vec<(String, Type)>),
    Generic(String),
    DataConstructor(String, Option<Box<Type>>, String),
    Unit,
    // compiler generated
    Any,
}

impl SizeOf for Type {
    fn size_of(&self) -> usize {
        let arch_size = if cfg!(target_pointer_width = "64") {
            8
        } else {
            4
        };
        use Type::*;
        match self {
            Int => 4,
            Usize => arch_size,
            Byte => 1,
            Bool => 1,
            Str => 0,
            TypeType => 0,
            OpaquePtr => arch_size,
            Ptr(_) => arch_size,
            Array(t) => t.size_of(),
            Tuple(types) => types.iter().map(|t| t.size_of()).sum(),
            Fn(_) => arch_size,
            // todo: look up size
            UserDef(_ident) => 0,
            Record(fields) => fields.iter().map(|(_, ty)| ty.size_of()).sum(),
            Generic(_) => 0,
            DataConstructor(_, data, _) => data.as_ref().map(|data| data.size_of()).unwrap_or(0), // todo: look up size
            Any => 0,
            Unit => 0,
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Type::*;
        match self {
            TypeType => write!(f, "type"),
            Int => write!(f, "int"),
            Usize => write!(f, "usize"),
            Bool => write!(f, "bool"),
            Str => write!(f, "str"),
            Generic(c) => write!(f, "'{}", c),
            Array(t) => write!(f, "[{}]", t),
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
                "Record {{{}}}",
                fields
                    .iter()
                    .map(|(name, ty)| format!("{}: {}", name, ty))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            DataConstructor(ident, data, _) => {
                if let Some(data) = data {
                    write!(f, "{}({})", ident, data)
                } else {
                    write!(f, "{}", ident)
                }
            }
            Unit => write!(f, "()"),
            OpaquePtr => write!(f, "opaqueptr"),
            Ptr(t) => write!(f, "*{}", t),
            Byte => write!(f, "byte"),
            Any => write!(f, "any"),
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
            Some((Token::BinOp(BinOp::Mul), span)) => {
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
                Ok((Type::Array(Box::new(ty)), span))
            }
            Some((Token::Tick, span)) => {
                parser.pop_next();
                let (c, span_end) = parser.expect_ident()?;
                let span = span.start..span_end.end;
                Ok((Type::Generic(c), span))
            }
            Some((Token::Keyword(Keyword::Type), span)) => {
                parser.pop_next();
                Ok((Type::TypeType, span))
            }
            Some((Token::Keyword(Keyword::Opaqueptr), span)) => {
                parser.pop_next();
                Ok((Type::OpaquePtr, span))
            }
            _ => {
                let (kw, span) = parser.expect_kw()?;
                match kw {
                    Keyword::Int => Ok((Type::Int, span)),
                    Keyword::Usize => Ok((Type::Usize, span)),
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

impl Type {
    pub fn instantiate_type(self, inst_types: &HashMap<String, Type>) -> Type {
        match self {
            Type::UserDef(ident) => {
                if let Some(ty) = inst_types.get(&ident) {
                    ty.clone()
                } else {
                    Type::UserDef(ident)
                }
            }
            Type::Ptr(ty) => {
                let inner_ty = ty.instantiate_type(inst_types);
                Type::Ptr(Box::new(inner_ty))
            }
            Type::Array(ty) => {
                let inner_ty = ty.instantiate_type(inst_types);
                Type::Array(Box::new(inner_ty))
            }
            Type::Tuple(types) => {
                let types = types
                    .into_iter()
                    .map(|t| t.instantiate_type(inst_types))
                    .collect();
                Type::Tuple(types)
            }
            Type::Record(fields) => {
                let fields = fields
                    .into_iter()
                    .map(|(name, ty)| (name, ty.instantiate_type(inst_types)))
                    .collect();
                Type::Record(fields)
            }
            ty => ty,
        }
    }
}
