use std::collections::HashMap;

use crate::{
    ScopedCtx, SizeOf, Span,
    ast::{
        Literal, Type,
        typed::{ExpressionKind, TypedExpression},
    },
};

fn size_of(ty: &Type) -> TypedExpression {
    let type_size = ty.size_of();
    println!("size of {} is {}", ty, type_size);
    TypedExpression {
        ty: Type::Int,
        kind: ExpressionKind::Literal(Literal::Int(type_size as i32)),
        span: Span::default(),
    }
}

fn str_len(expr: TypedExpression) -> TypedExpression {
    let s = match expr.kind {
        ExpressionKind::Literal(Literal::Str(s)) => s,
        _ => panic!("got {expr:?}, expected str literal"),
    };

    TypedExpression {
        ty: Type::Int,
        kind: ExpressionKind::Literal(Literal::Int(s.len() as i32)),
        span: expr.span,
    }
}

fn cast(ty: TypedExpression, mut expr: TypedExpression) -> TypedExpression {
    let ty = match ty.kind {
        ExpressionKind::Type(ty) => ty,
        _ => panic!("got {ty:?}, expected type"),
    };
    expr.ty = ty;
    expr
}

// this takes an intrinsic and it's args, and then maps it to a typed expression
pub fn resolve_intrinsic(
    name: &str,
    mut args: Vec<TypedExpression>,
    instantiated_types: &HashMap<String, Type>,
    _ctx: &ScopedCtx,
) -> TypedExpression {
    match name {
        "sizeOf" => {
            let arg = args.remove(0);
            let ty_name = match arg.kind {
                ExpressionKind::Type(Type::UserDef(ty_name)) => ty_name,
                ExpressionKind::Ident(ty_name) => ty_name,
                _ => panic!("got {arg:?}, expected type"),
            };
            let ty = instantiated_types
                .get(&ty_name)
                .unwrap_or_else(|| panic!("type {} not found", ty_name));
            size_of(ty)
        }
        "strLen" => str_len(args.remove(0)),
        "cast" => {
            let ty = args.remove(0);
            let expr = args.remove(0);
            cast(ty, expr)
        }
        i => panic!("intrinsic {} not found", i),
    }
}

pub fn intrinsic_type(name: &str) -> Type {
    match name {
        "sizeOf" => Type::Int,
        "strLen" => Type::Int,
        "cast" => Type::Any,
        i => panic!("intrinsic {} not found", i),
    }
}

// determines if an intrinsic is resolveable instantly
pub fn has_comptime_args(name: &str) -> bool {
    match name {
        "sizeOf" => true,
        "strLen" => false,
        "cast" => true,
        _ => false,
    }
}
