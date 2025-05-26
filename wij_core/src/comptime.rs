use crate::{
    SizeOf, Span,
    ast::{
        Literal, Type,
        typed::{ExpressionKind, TypedExpression},
    },
};

fn size_of(ty: Type) -> TypedExpression {
    let type_size = ty.size_of();
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
pub fn resolve_intrinsic(name: &str, mut args: Vec<TypedExpression>) -> TypedExpression {
    match name {
        "sizeOf" => {
            let ty = args[0].ty.clone();
            size_of(ty)
        }
        "cast" => cast(args.swap_remove(0)),
        i => panic!("intrinsic {} not found", i),
    }
}
