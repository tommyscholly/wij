use std::collections::HashMap;

use crate::{
    SizeOf, Span,
    ast::{
        Literal, Type,
        typed::{ExpressionKind, TypedExpression},
    },
};

fn size_of(ty: &Type) -> TypedExpression {
    let type_size = ty.size_of();
    println!("size of {} is {}", ty, type_size);
    TypedExpression {
        ty: Type::Usize,
        kind: ExpressionKind::Literal(Literal::Usize(type_size)),
        span: Span::default(),
    }
}

fn cast(mut expr: TypedExpression) -> TypedExpression {
    expr.ty = Type::Any;
    expr
}

// this takes an intrinsic and it's args, and then maps it to a typed expression
pub fn resolve_intrinsic(
    name: &str,
    mut args: Vec<TypedExpression>,
    instantiated_types: &HashMap<String, Type>,
) -> TypedExpression {
    match name {
        "sizeOf" => {
            let arg = args.remove(0);
            let ty_name = match arg.kind {
                ExpressionKind::Type(Type::UserDef(ty_name)) => ty_name,
                ExpressionKind::Ident(ty_name) => ty_name,
                _ => panic!("got {arg:?}, expected type"),
            };
            let ty = instantiated_types.get(&ty_name).expect("ty not found");
            size_of(ty)
        }
        "cast" => cast(args.swap_remove(0)),
        i => panic!("intrinsic {} not found", i),
    }
}

pub fn intrinsic_type(name: &str) -> Type {
    match name {
        "sizeOf" => Type::Usize,
        "cast" => Type::Any,
        i => panic!("intrinsic {} not found", i),
    }
}
