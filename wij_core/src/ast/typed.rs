use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    mem,
};

use crate::{
    WijError,
    ast::{self, Type},
    comptime,
};

use super::{
    BinOp, DeclKind as ASTDeclKind, Declaration, Expression, Function, Literal, Span, Spanned,
    Statement, Var,
};

#[derive(Debug)]
pub enum TypeErrorKind {
    NoReturn,
    UndefinedVariable(String),
    UndefinedType(String),
    UndefinedFunction(String),
    Unassignable,
    IdentUsedAsFn(String),
    DuplicateModule,
    ModuleNotFound,
    InvalidType(Type),
    FunctionArityMismatch {
        expected: u32,
        found: u32,
    },
    MismatchComptimeArgs {
        expected: usize,
        found: usize,
    },
    TypeAnnotationNeeded,
    TypeMismatch {
        expected: Type,
        found: Type,
    },
    IncompatibleTypes {
        operation: BinOp,
        lhs: Type,
        rhs: Type,
    },
}

impl Display for TypeErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use TypeErrorKind::*;
        match self {
            MismatchComptimeArgs { expected, found } => {
                write!(
                    f,
                    "Expected {} comptime args, but found {}",
                    expected, found
                )
            }
            TypeAnnotationNeeded => write!(f, "Type annotation needed"),
            NoReturn => write!(f, "No return statement"),
            InvalidType(ty) => write!(f, "Invalid type `{}`", ty),
            UndefinedVariable(ident) => write!(f, "Undefined variable `{}`", ident),
            UndefinedType(ident) => write!(f, "Undefined type `{}`", ident),
            UndefinedFunction(ident) => write!(f, "Undefined function `{}`", ident),
            Unassignable => write!(f, "Cannot assign to this expression"),
            IdentUsedAsFn(ident) => write!(f, "Identifier `{}` used as function", ident),
            FunctionArityMismatch { expected, found } => {
                write!(f, "Expected {} arguments, but found {}", expected, found)
            }
            TypeMismatch { expected, found } => {
                write!(f, "Expected `{}` but found `{}`", expected, found)
            }
            IncompatibleTypes {
                operation,
                lhs,
                rhs,
            } => write!(
                f,
                "Incompatible types for operation `{}`: {} and {}",
                operation, lhs, rhs
            ),
            DuplicateModule => write!(f, "Duplicate module"),
            ModuleNotFound => write!(f, "Module not found"),
        }
    }
}

#[derive(Debug)]
pub struct TypeError {
    kind: TypeErrorKind,
    notes: Vec<(String, Span)>,
    span: Span,
}

impl TypeError {
    pub fn new(kind: TypeErrorKind, span: Span) -> Self {
        Self {
            kind,
            span,
            notes: Vec::new(),
        }
    }

    pub fn add_note(mut self, note: String, span: Span) -> Self {
        self.notes.push((note, span));
        self
    }
}

impl WijError for TypeError {
    fn span(&self) -> Option<Span> {
        Some(self.span.clone())
    }

    fn reason(&self) -> String {
        self.kind.to_string()
    }

    fn notes(&self) -> Vec<(String, Span)> {
        self.notes.clone()
    }
}

pub type TypeResult<T> = Result<T, TypeError>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FunctionSignature {
    pub param_types: Vec<Type>,
    pub ret_type: Type,
}

impl Display for FunctionSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) -> {}",
            self.param_types
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<String>>()
                .join(", "),
            self.ret_type
        )
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypedVar {
    pub id: String,
    pub ty: Type,
    pub is_comptime: bool,
    pub span: Span,
}

trait Expressions {
    fn expressions(&mut self) -> Vec<&mut TypedExpression>;
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Intrinsic {
    pub name: String,
    pub args: Vec<TypedExpression>,
}

impl Intrinsic {
    fn resolve(self) -> TypedExpression {
        comptime::resolve_intrinsic(&self.name, self.args)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum StatementKind {
    Let {
        var: TypedVar,
        value: Option<TypedExpression>,
    },
    Return(Option<TypedExpression>),
    Break,
    Continue,
    // the optional type is the return type of the block
    Block(Vec<TypedStatement>),
    If {
        condition: TypedExpression,
        then_block: Box<TypedStatement>,
        else_block: Option<Box<TypedStatement>>,
    },
    Assignment(Box<TypedExpression>, Box<TypedExpression>),
    For {
        var: TypedVar,
        in_expr: TypedExpression,
        body: Box<TypedStatement>,
    },
    While {
        condition: TypedExpression,
        body: Box<TypedStatement>,
    },
    // Match {
    //     value: TypedExpression,
    //     cases: Vec<MatchCase>,
    // },
    Intrinsic(Intrinsic),
    Expression(Box<TypedExpression>), // a subclass of expressions being executed for side effects
}

impl Expressions for StatementKind {
    fn expressions(&mut self) -> Vec<&mut TypedExpression> {
        match self {
            StatementKind::Block(stmts) => stmts.iter_mut().flat_map(|s| s.expressions()).collect(),
            StatementKind::If {
                condition,
                then_block,
                else_block,
            } => {
                let mut exprs = then_block.expressions();
                exprs.push(condition);
                if let Some(b) = else_block {
                    exprs.extend(b.expressions());
                }
                exprs
            }
            StatementKind::For {
                var: _,
                in_expr: _,
                body,
            } => body.expressions(),
            StatementKind::Let { value, .. } => value.as_mut().map_or(vec![], |e| vec![e]),
            StatementKind::Assignment(lhs, rhs) => vec![lhs.as_mut(), rhs.as_mut()],
            StatementKind::Return(ty_expr) => ty_expr.as_mut().map_or(vec![], |e| vec![e]),
            StatementKind::While { condition, body } => {
                let mut exprs = body.expressions();
                exprs.push(condition);
                exprs
            }
            StatementKind::Expression(expr) => vec![expr.as_mut()],
            StatementKind::Break | StatementKind::Continue => vec![],
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypedStatement {
    pub kind: StatementKind,
    pub ty: Type,
    pub span: Span,
}

impl Expressions for TypedStatement {
    fn expressions(&mut self) -> Vec<&mut TypedExpression> {
        self.kind.expressions()
    }
}

impl TypedStatement {
    fn has_return(&self) -> Option<Type> {
        match &self.kind {
            StatementKind::Return(ty_expr) => match ty_expr {
                Some(ty_expr) => Some(ty_expr.ty.clone()),
                None => Some(Type::Unit),
            },
            StatementKind::Block(stmts) => stmts.iter().find_map(|stmt| stmt.has_return()),
            StatementKind::If {
                condition: _,
                then_block,
                else_block,
            } => {
                let then_block_ret = then_block.has_return();
                let else_block_ret = else_block.as_ref().and_then(|b| b.has_return());

                match (then_block_ret, else_block_ret) {
                    (Some(ty), None) => Some(ty),
                    (None, Some(ty)) => Some(ty),
                    (Some(ty), Some(ty2)) => {
                        if ty == ty2 {
                            Some(ty)
                        } else {
                            panic!("todo: fill in mismatching return types")
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ExpressionKind {
    Literal(Literal),
    Ident(String),
    Array(Vec<TypedExpression>),
    FieldAccess(Box<TypedExpression>, String),
    BinOp(BinOp, Box<TypedExpression>, Box<TypedExpression>),
    FnCall(String, Vec<TypedExpression>),
    RecordInit(String, Vec<(String, TypedExpression)>),
    DataConstructor(String, Option<Box<TypedExpression>>),
    Idx(Box<TypedExpression>, Box<TypedExpression>),
    // this is for comptime intrinsics
    // these should all be erased by SSA conversion
    Type(Type),
    Self_,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypedExpression {
    pub kind: ExpressionKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct EnumVariant {
    pub name: String,
    pub data: Option<Spanned<Type>>,
    pub span: Span,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum DeclKind {
    Function {
        name: String,
        arguments: Vec<TypedVar>,
        body: TypedStatement,
        ret_type: Option<Type>,
    },
    ComptimeFunction {
        name: String,
        comptime_args: Vec<String>,
        arguments: Vec<TypedVar>,
        body: TypedStatement,
        ret_type: Option<Type>,
    },
    Record {
        name: String,
        fields: Vec<(String, Type)>,
    },
    Enum {
        name: String,
        variants: Vec<EnumVariant>,
    },
    ForeignDeclarations(Vec<(String, FunctionSignature)>),
    // We implement procedures on a type
    Procedures(Type, Vec<TypedDecl>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypedDecl {
    pub kind: DeclKind,
    pub ty: Type,
    pub span: Span,
    pub visible: bool,
}

impl TypedDecl {
    pub fn name(&self) -> Option<&str> {
        match &self.kind {
            DeclKind::Function { name, .. } => Some(name),
            DeclKind::Record { name, .. } => Some(name),
            DeclKind::Enum { name, .. } => Some(name),
            DeclKind::ForeignDeclarations(..) => None,
            // todo: evaluate if we want to define procedures with the name idea in
            // `register_types`
            DeclKind::Procedures(..) => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Module {
    pub name: String,
    pub decls: Vec<TypedDecl>,
    // todo: idxs of decls that are exported?
    pub exports: Vec<TypedDecl>,
    pub comptime_exports: Vec<Spanned<Declaration>>,
    pub instantiated_comptime_decls: Vec<TypedDecl>,
}

impl Module {
    pub fn new(name: String) -> Module {
        Module {
            name,
            decls: vec![],
            exports: vec![],
            comptime_exports: vec![],
            instantiated_comptime_decls: vec![],
        }
    }

    pub fn combine(&mut self, mut other: Module) {
        let mut new_decls = Vec::new();
        // prepend the other decls so imports take precedence
        new_decls.append(&mut other.decls);
        new_decls.append(&mut self.decls);
        self.decls = new_decls;
        self.exports.append(&mut other.exports);
        self.comptime_exports.append(&mut other.comptime_exports);
        self.instantiated_comptime_decls
            .append(&mut other.instantiated_comptime_decls);
    }
}

#[derive(Default, Clone, Debug)]
pub struct TyCtx {
    comptime_fns: HashSet<String>,
    var_id_type: HashMap<String, Type>,
    user_def_types: HashMap<String, Type>,
}

impl TyCtx {
    pub fn new() -> TyCtx {
        TyCtx::default()
    }

    fn insert_user_def_type(&mut self, name: String, ty: Type) {
        self.user_def_types.insert(name, ty);
    }

    fn insert_comptime_fn(&mut self, name: String, ty: Type) {
        self.comptime_fns.insert(name.clone());
        self.user_def_types.insert(name, ty);
    }

    pub fn get_user_def_type(&self, name: &str) -> Option<Type> {
        self.user_def_types.get(name).cloned()
    }
}

#[derive(Debug, Clone)]
pub struct ScopedCtx<'a> {
    ctx: TyCtx,
    parent: Option<&'a ScopedCtx<'a>>,
}

impl<'a> ScopedCtx<'a> {
    fn from_tyctx(ctx: TyCtx) -> ScopedCtx<'a> {
        ScopedCtx { ctx, parent: None }
    }

    fn child(&'a self) -> ScopedCtx<'a> {
        ScopedCtx {
            ctx: TyCtx::new(),
            parent: Some(self),
        }
    }

    fn var_ty(&self, id: &str) -> Option<&Type> {
        match self.ctx.var_id_type.get(id) {
            Some(ty) => Some(ty),
            None => match self.parent {
                Some(parent) => parent.var_ty(id),
                None => None,
            },
        }
    }

    fn insert_comptime_fn(&mut self, name: String, ty: Type) {
        self.ctx.insert_comptime_fn(name, ty);
    }

    fn insert_var(&mut self, name: String, ty: Type) {
        self.ctx.var_id_type.insert(name, ty);
    }

    fn get_user_def_type(&self, name: &str) -> Option<Type> {
        match self.ctx.get_user_def_type(name) {
            Some(ty) => Some(ty),
            None => match self.parent {
                Some(parent) => parent.get_user_def_type(name),
                None => None,
            },
        }
    }

    fn insert_user_def_type(&mut self, name: String, ty: Type) {
        self.ctx.user_def_types.insert(name, ty);
    }

    fn resolve_type(&self, ty: Type) -> Type {
        if let Type::UserDef(name) = &ty {
            return match self.get_user_def_type(name) {
                Some(ty) => ty,
                None => match self.parent {
                    Some(parent) => parent.resolve_type(ty),
                    None => ty,
                },
            };
        }

        ty
    }
}

fn type_var(ctx: &mut ScopedCtx, var: Spanned<Var>, inferred_ty: Type) -> TypeResult<TypedVar> {
    let (mut var, span) = var;
    match (&var.ty, inferred_ty) {
        (Some(_), Type::Any) => {
            // do nothing
        }
        (Some(ty), ity) => {
            let ty = ctx.resolve_type(ty.clone());
            let ity = ctx.resolve_type(ity);
            if ty != ity {
                return Err(TypeError::new(
                    TypeErrorKind::TypeMismatch {
                        expected: ty.clone(),
                        found: ity.clone(),
                    },
                    span,
                ));
            }
        }
        (None, Type::Any) => {
            return Err(TypeError::new(TypeErrorKind::TypeAnnotationNeeded, span));
        }
        (None, ty) => {
            var.ty = Some(ty);
        }
    };

    ctx.insert_var(var.name.clone(), var.ty.clone().unwrap());
    Ok(TypedVar {
        id: var.name,
        ty: var.ty.unwrap(),
        is_comptime: var.is_comptime,
        span,
    })
}

// Filters decls into decls, and modules
// Should only ever be one module
fn extract_module(
    decls: Vec<Spanned<Declaration>>,
) -> TypeResult<(Module, Vec<Spanned<Declaration>>)> {
    let mut filt_decls: Vec<Spanned<Declaration>> = vec![];
    let mut modules = vec![];
    for decl in decls {
        match decl.0.decl {
            ASTDeclKind::Module(name) => {
                modules.push((name, decl.1));
            }
            ASTDeclKind::Use(_) => {}
            _ => {
                filt_decls.push(decl);
            }
        }
    }

    if modules.len() > 1 {
        return Err(TypeError::new(
            TypeErrorKind::DuplicateModule,
            modules[0].1.start..modules[1].1.end,
        ));
    } else if modules.is_empty() {
        return Err(TypeError::new(
            TypeErrorKind::ModuleNotFound,
            Span::default(),
        ));
    }

    let module = Module {
        name: modules[0].0.clone(),
        decls: vec![],
        exports: vec![],
        comptime_exports: vec![],
        instantiated_comptime_decls: vec![],
    };

    Ok((module, filt_decls))
}

pub fn extract_comptime_fns(
    decls: Vec<Spanned<Declaration>>,
) -> (Vec<Spanned<Declaration>>, Vec<Spanned<Declaration>>) {
    let mut comptime_fns = vec![];
    let mut filt_decls = vec![];

    for decl in decls {
        match &decl.0.decl {
            ASTDeclKind::Function(Function { arguments, .. }) => {
                let has_comptime = arguments.iter().any(|(var, _)| var.is_comptime);
                if has_comptime {
                    comptime_fns.push(decl);
                } else {
                    filt_decls.push(decl);
                }
            }
            _ => filt_decls.push(decl),
        }
    }

    (comptime_fns, filt_decls)
}

// fn comptime_fn_calls<'a>(
//     ctx: &mut ScopedCtx,
//     decl: &mut TypedDecl,
//     comptime_fns: HashMap<String, Spanned<Declaration>>,
// ) -> Vec<TypedDecl> {
//     let DeclKind::Function { body, .. } = &mut decl.kind else {
//         return vec![];
//     };
//
//     let expressions = body.expressions();
//     if expressions.is_empty() {
//         return vec![];
//     }
//
//     let mut monomorphic_fns = vec![];
//     for expr in expressions {
//         if let ExpressionKind::FnCall(fn_name, args) = &expr.kind {
//             if let Some(decl) = comptime_fns.get(fn_name) {
//                 todo!()
//             }
//         }
//     }
//
//     monomorphic_fns
// }

pub struct TypeChecker<'a> {
    decls: Vec<Spanned<Declaration>>,
    module: Module,
    imports: Vec<TypedDecl>,
    comptime_fns: HashMap<String, Spanned<Function>>,
    ty_decls: Vec<TypedDecl>,
    monomorphic_fns: HashMap<String, TypedDecl>,
    top_ctx: *mut ScopedCtx<'a>,
}

impl<'a> TypeChecker<'a> {
    pub fn new(
        decls: Vec<Spanned<Declaration>>,
        imports: Vec<TypedDecl>,
        comptime_imports: Vec<Spanned<Declaration>>,
    ) -> TypeResult<Self> {
        let (mut module, decls) = extract_module(decls)?;
        let (mut comptime_fns, decls) = extract_comptime_fns(decls);
        comptime_fns.extend(comptime_imports);
        let mut comptime_map = HashMap::new();
        for (decl, span) in comptime_fns {
            let name = decl.name().unwrap().to_string();
            if decl.visibility.to_bool() {
                module.comptime_exports.push((decl.clone(), span.clone()));
            }

            let declfn = match decl.decl {
                ASTDeclKind::Function(fn_) => fn_,
                _ => unreachable!(),
            };

            comptime_map.insert(name, (declfn, span));
        }

        let ctx = TyCtx::new();
        let scoped_ctx = ScopedCtx::from_tyctx(ctx);
        let tychkr = TypeChecker {
            module,
            decls,
            imports,
            ty_decls: vec![],
            monomorphic_fns: HashMap::new(),
            comptime_fns: comptime_map,
            top_ctx: Box::into_raw(Box::new(scoped_ctx)),
        };

        Ok(tychkr)
    }

    fn register_types(&self, ctx: &mut ScopedCtx, imports: &Vec<TypedDecl>) {
        let decls = &self.decls;
        for decl in decls {
            match &decl.0.decl {
                ASTDeclKind::Record { name, fields } => {
                    let fields: Vec<(String, Type)> = fields
                        .iter()
                        .map(|(var, _)| (var.name.clone(), var.ty.clone().unwrap()))
                        .collect();
                    let type_ = Type::Record(fields);
                    ctx.insert_user_def_type(name.to_string(), type_);
                }
                ASTDeclKind::Enum {
                    name: enum_name,
                    variants,
                } => {
                    for (variant, _) in variants {
                        let ast::EnumVariant { name, data, .. } = variant;
                        let data = data.as_ref().map(|data| Box::new(data.0.clone()));
                        ctx.insert_user_def_type(
                            name.to_string(),
                            Type::DataConstructor(name.to_string(), data, enum_name.to_string()),
                        );
                    }
                }
                _ => {}
            }
        }

        let decls = decls.clone();
        for decl in decls {
            let (name, arguments, _body, ret_type, num_comptime_args) = match &decl.0.decl {
                ASTDeclKind::Function(Function {
                    name,
                    arguments,
                    num_comptime_args,
                    body,
                    ret_type,
                }) => (name, arguments, body, ret_type, num_comptime_args),
                ASTDeclKind::Procedures(type_name, fns) => {
                    for fn_def in fns {
                        let Function {
                            name,
                            arguments,
                            num_comptime_args: _,
                            body: _,
                            ret_type,
                        } = &fn_def.0;
                        let mut param_types = vec![type_name.clone()];

                        for (var, _) in arguments {
                            param_types.push(var.ty.clone().unwrap());
                        }

                        let name = format!("{}::{}", type_name, name);
                        let signature = FunctionSignature {
                            param_types,
                            ret_type: ret_type.clone().unwrap_or(Type::Unit),
                        };

                        println!("registered method fn {}: {}", name, signature);
                        ctx.insert_user_def_type(name, Type::Fn(Box::new(signature)));
                    }
                    continue;
                }
                _ => continue,
            };

            let param_types: Vec<Type> = arguments
                .iter()
                .map(|(var, _)| var.ty.clone().unwrap())
                .collect();

            let ret_type = ret_type.clone().unwrap_or(Type::Unit);
            let signature = FunctionSignature {
                param_types,
                ret_type,
            };
            println!("registered fn {}: {}", name, signature);
            if *num_comptime_args > 0 {
                ctx.insert_comptime_fn(name.to_string(), Type::Fn(Box::new(signature)));
            } else {
                ctx.insert_user_def_type(name.to_string(), Type::Fn(Box::new(signature)));
            }
        }

        for decl in imports {
            if let Some(name) = decl.name() {
                ctx.insert_user_def_type(name.to_string(), decl.ty.clone());
            }

            if let DeclKind::Procedures(_, fns) = &decl.kind {
                for proc_type_decl in fns {
                    ctx.insert_user_def_type(
                        proc_type_decl.name().unwrap().to_string(),
                        proc_type_decl.ty.clone(),
                    );
                }
            }
        }
    }

    pub fn produce_module(mut self) -> TypeResult<Module> {
        let ctx = unsafe { &mut *self.top_ctx };
        self.register_types(ctx, &self.imports);
        let mut exports = vec![];
        let decls = mem::take(&mut self.decls);
        for decl in decls.into_iter() {
            let ty_decl = self.type_decl(ctx, decl)?;
            if ty_decl.visible {
                exports.push(ty_decl.clone());
            }
            self.ty_decls.push(ty_decl);
        }

        println!("ty decls: {:#?}", self.ty_decls);
        self.module.decls = self.ty_decls;
        self.module.exports = exports;

        Ok(self.module)
    }

    fn type_decl(
        &mut self,
        ctx: &mut ScopedCtx,
        decl: Spanned<Declaration>,
    ) -> TypeResult<TypedDecl> {
        let decl = match decl.0.decl {
            ASTDeclKind::Procedures(ty, procs) => {
                let mut ty_procs = vec![];
                for (mut proc_fn, span) in procs {
                    // kind of a hack, changing the method name to be a function that just exists with
                    // the name type::name
                    proc_fn.name = format!("{}::{}", ty, proc_fn.name);
                    let ty_fn =
                        self.type_fn(ctx, proc_fn, span, Some(ty.clone()), HashMap::new())?;
                    ty_procs.push(ty_fn);
                }
                TypedDecl {
                    ty: ty.clone(),
                    kind: DeclKind::Procedures(ty, ty_procs),
                    span: decl.1,
                    visible: true,
                }
            }
            ASTDeclKind::Function(fn_def) => {
                let mut fn_decl = self.type_fn(ctx, fn_def, decl.1, None, HashMap::new())?;
                fn_decl.visible = decl.0.visibility.to_bool();
                fn_decl
            }
            ASTDeclKind::Enum { name, variants } => TypedDecl {
                ty: Type::Unit,
                kind: DeclKind::Enum {
                    name,
                    variants: variants
                        .into_iter()
                        .map(|var| EnumVariant {
                            name: var.0.name,
                            data: var.0.data,
                            span: var.1,
                        })
                        .collect(),
                },
                span: decl.1,
                visible: decl.0.visibility.to_bool(),
            },
            ASTDeclKind::Record { name, fields } => {
                let fields: Vec<(String, Type)> = fields
                    .into_iter()
                    .map(|(var, _)| (var.name, var.ty.unwrap()))
                    .collect();

                TypedDecl {
                    ty: ctx
                        .get_user_def_type(&name)
                        .expect("record type should have been parsed"),
                    kind: DeclKind::Record { name, fields },
                    span: decl.1,
                    visible: decl.0.visibility.to_bool(),
                }
            }
            ASTDeclKind::ForeignDeclarations(fds) => {
                let mut ty_fds = vec![];
                for (fd, _) in fds {
                    let name = fd.name;
                    let sig = fd.sig;
                    ctx.insert_user_def_type(name.clone(), Type::Fn(Box::new(sig.clone())));
                    ty_fds.push((name, sig));
                }

                TypedDecl {
                    ty: Type::Unit,
                    kind: DeclKind::ForeignDeclarations(ty_fds),
                    span: decl.1,
                    visible: decl.0.visibility.to_bool(),
                }
            }
            ASTDeclKind::Use(_) => todo!(),
            ASTDeclKind::Module(_) => unreachable!(),
        };

        Ok(decl)
    }

    fn type_stmt(
        &mut self,
        ctx: &mut ScopedCtx,
        stmt: Spanned<Statement>,
    ) -> TypeResult<TypedStatement> {
        let stmt = match stmt.0 {
            Statement::Break => TypedStatement {
                kind: StatementKind::Break,
                ty: Type::Unit,
                span: stmt.1,
            },
            Statement::Continue => TypedStatement {
                kind: StatementKind::Continue,
                ty: Type::Unit,
                span: stmt.1,
            },
            Statement::For { var, in_expr, body } => {
                let ty_in_expr = self.type_expr(ctx, in_expr)?;
                let var_ty = if let Type::Array(inner_ty) = &ty_in_expr.ty {
                    *inner_ty.clone()
                } else {
                    ty_in_expr.ty.clone()
                };

                let mut child_ctx = ctx.child();
                let ty_var = type_var(&mut child_ctx, var, var_ty)?;
                let ty_body = self.type_stmt(&mut child_ctx, *body)?;

                TypedStatement {
                    kind: StatementKind::For {
                        var: ty_var,
                        in_expr: ty_in_expr,
                        body: Box::new(ty_body),
                    },
                    ty: Type::Unit,
                    span: stmt.1,
                }
            }
            Statement::While { condition, body } => {
                let ty_condition = self.type_expr(ctx, condition)?;

                let mut child_ctx = ctx.child();
                let ty_body = self.type_stmt(&mut child_ctx, *body)?;

                TypedStatement {
                    kind: StatementKind::While {
                        condition: ty_condition,
                        body: Box::new(ty_body),
                    },
                    ty: Type::Unit,
                    span: stmt.1,
                }
            }
            Statement::Assignment(lhs, rhs) => {
                let ty_lhs = self.type_expr(ctx, *lhs)?;
                let ty_rhs = self.type_expr(ctx, *rhs)?;

                if ty_lhs.ty != ty_rhs.ty {
                    let err = TypeError::new(
                        TypeErrorKind::TypeMismatch {
                            expected: ty_lhs.ty.clone(),
                            found: ty_rhs.ty.clone(),
                        },
                        stmt.1,
                    )
                    .add_note(format!("This has type {}", ty_lhs.ty), ty_lhs.span)
                    .add_note(format!("This has type {}", ty_rhs.ty), ty_rhs.span);
                    return Err(err);
                }

                if !matches!(
                    ty_lhs.kind,
                    ExpressionKind::Ident(_)
                        | ExpressionKind::FieldAccess(_, _)
                        | ExpressionKind::Idx(_, _)
                ) {
                    return Err(TypeError::new(TypeErrorKind::Unassignable, stmt.1));
                }

                TypedStatement {
                    ty: ty_lhs.ty.clone(),
                    kind: StatementKind::Assignment(Box::new(ty_lhs), Box::new(ty_rhs)),
                    span: stmt.1,
                }
            }
            Statement::Let { var, value } => {
                // TODO: cleanup these clones
                let (val, inferred_ty) = if let Some(expr) = value {
                    let ty_expr = self.type_expr(ctx, expr)?;
                    let inferred_ty = ty_expr.ty.clone();
                    (Some(ty_expr), inferred_ty)
                } else {
                    (None, Type::Any)
                };

                let var_ty = var.0.ty.clone();
                let ty_var = match type_var(ctx, var, inferred_ty) {
                    Ok(ty_var) => ty_var,
                    Err(err) => match val {
                        Some(val) => {
                            return Err(TypeError::new(
                                TypeErrorKind::TypeMismatch {
                                    expected: var_ty.clone().unwrap(),
                                    found: val.ty.clone(),
                                },
                                stmt.1,
                            )
                            .add_note(format!("This has type {}", var_ty.unwrap()), err.span)
                            .add_note(format!("This has type {}", val.ty), val.span));
                        }
                        None => return Err(err),
                    },
                };

                TypedStatement {
                    kind: StatementKind::Let {
                        var: ty_var,
                        value: val,
                    },
                    ty: Type::Unit,
                    span: stmt.1,
                }
            }
            Statement::Return(value) => {
                let val = if let Some(expr) = value {
                    Some(self.type_expr(ctx, expr)?)
                } else {
                    None
                };

                TypedStatement {
                    kind: StatementKind::Return(val),
                    ty: Type::Unit,
                    span: stmt.1,
                }
            }
            Statement::Block(stmts) => {
                let mut ty_stmts = vec![];
                let mut child_ctx = ctx.child();

                for stmt in stmts {
                    let ty_stmt = self.type_stmt(&mut child_ctx, stmt)?;

                    ty_stmts.push(ty_stmt);
                }

                TypedStatement {
                    kind: StatementKind::Block(ty_stmts),
                    ty: Type::Unit,
                    span: stmt.1,
                }
            }
            Statement::If {
                condition,
                then_block,
                else_block,
            } => {
                let mut else_ctx = ctx.child();
                let else_block = if let Some(stmt) = else_block {
                    Some(Box::new(self.type_stmt(&mut else_ctx, *stmt)?))
                } else {
                    None
                };

                let mut then_ctx = ctx.child();
                let then_block = Box::new(self.type_stmt(&mut then_ctx, *then_block)?);
                TypedStatement {
                    kind: StatementKind::If {
                        condition: self.type_expr(ctx, condition)?,
                        then_block,
                        else_block,
                    },
                    ty: Type::Unit,
                    span: stmt.1,
                }
            }
            Statement::Expression(expr) => {
                let expr = self.type_expr(ctx, *expr)?;
                TypedStatement {
                    ty: expr.ty.clone(),
                    kind: StatementKind::Expression(Box::new(expr)),
                    span: stmt.1,
                }
            }

            #[allow(unused)]
            Statement::Match { value, cases } => todo!(),
        };

        Ok(stmt)
    }

    pub fn type_fn(
        &mut self,
        ctx: &mut ScopedCtx,
        fn_def: Function,
        span: Span,
        self_arg: Option<Type>,
        instantiated_comptime_types: HashMap<String, Type>,
    ) -> TypeResult<TypedDecl> {
        let Function {
            name,
            arguments,
            num_comptime_args,
            body,
            ret_type,
        } = fn_def;
        let mut body_ctx = ctx.child();
        let mut args = vec![];
        if let Some(self_ty) = self_arg {
            let ty_var = TypedVar {
                id: "self".to_string(),
                ty: self_ty.clone(),
                is_comptime: false,
                // span does not matter
                span: Span::default(),
            };
            body_ctx.insert_var("self".to_string(), self_ty);
            args.push(ty_var);
        }

        let mut fn_is_comptime = false;
        let mut comptime_args = vec![];

        for var in arguments {
            let ity = var.0.ty.clone().unwrap();
            let ty_var = type_var(&mut body_ctx, var, ity)?;
            if ty_var.is_comptime {
                fn_is_comptime = true;
                body_ctx.insert_var(
                    ty_var.id.clone(),
                    instantiated_comptime_types
                        .get(&ty_var.id)
                        .unwrap_or_else(|| {
                            panic!("{} not found in instantiated_comptime_types {instantiated_comptime_types:?}", ty_var.id)
                        })
                        .clone(),
                );
                comptime_args.push(ty_var);
            } else {
                args.push(ty_var);
            }
        }
        println!("args: {args:?}");
        println!("cmptime: {comptime_args:?}");

        if comptime_args.len() != num_comptime_args {
            return Err(TypeError::new(
                TypeErrorKind::MismatchComptimeArgs {
                    expected: num_comptime_args,
                    found: comptime_args.len(),
                },
                span,
            ));
        }

        let name = if fn_is_comptime {
            println!("is comptime");
            let mut comp_time_arg_names = String::new();
            for arg in &comptime_args {
                let name = instantiated_comptime_types
                    .get(&arg.id)
                    .unwrap()
                    .to_string();
                comp_time_arg_names.push_str(&name);
            }
            let instantiated_fn_name = format!("{}_{}", name, comp_time_arg_names);
            if self.monomorphic_fns.contains_key(&instantiated_fn_name) {
                // todo: improve this
                return Ok(self
                    .monomorphic_fns
                    .get(&instantiated_fn_name)
                    .unwrap()
                    .clone());
            }
            instantiated_fn_name
        } else {
            name
        };

        println!("typing stmt");
        let body = self.type_stmt(&mut body_ctx, body)?;
        println!("typed stmt");
        let ret = match body.has_return() {
            Some(ty) => ty.instantiate_type(&instantiated_comptime_types),
            None => {
                if ret_type.is_some() {
                    return Err(TypeError::new(TypeErrorKind::NoReturn, body.span));
                }

                Type::Unit
            }
        };
        println!("ret: {}", ret);

        if !(ret == Type::Unit) && ret_type.is_none() {
            return Err(TypeError::new(
                TypeErrorKind::TypeMismatch {
                    expected: Type::Unit,
                    found: ret.clone(),
                },
                body.span,
            ));
        } else {
            #[allow(clippy::single_match)]
            match ret_type.as_ref() {
                Some(ret_type) => {
                    if ret != *ret_type {
                        return Err(TypeError::new(
                            TypeErrorKind::TypeMismatch {
                                expected: ret_type.clone(),
                                found: ret.clone(),
                            },
                            body.span,
                        ));
                    }
                }
                None => (),
            }
        }

        let ty_decl = TypedDecl {
            ty: match ctx.get_user_def_type(&name) {
                Some(ty) => ty,
                None => return Err(TypeError::new(TypeErrorKind::IdentUsedAsFn(name), span)),
            },
            kind: DeclKind::Function {
                name: name.clone(),
                arguments: args,
                body,
                ret_type,
                is_comptime: fn_is_comptime,
            },
            span,
            visible: false,
        };
        println!("out of tyfn for {name}");
        if fn_is_comptime {
            self.monomorphic_fns.insert(name, ty_decl.clone());
        }
        Ok(ty_decl)
    }

    fn type_expr(
        &mut self,
        ctx: &mut ScopedCtx,
        expr: Spanned<Expression>,
    ) -> TypeResult<TypedExpression> {
        let expr = match expr.0 {
            Expression::Type(ty) => {
                let ty = ctx.resolve_type(ty);
                TypedExpression {
                    ty: Type::TypeType,
                    kind: ExpressionKind::Type(ty),
                    span: expr.1,
                }
            }
            Expression::Self_ => {
                let ty = match ctx.var_ty("self") {
                    Some(ty) => ty.clone(),
                    None => {
                        return Err(TypeError::new(
                            TypeErrorKind::UndefinedVariable("self".to_string()),
                            expr.1,
                        ));
                    }
                };
                TypedExpression {
                    ty: ty.clone(),
                    kind: ExpressionKind::Self_,
                    span: expr.1,
                }
            }
            Expression::Intrinsic(name, args) => {
                // so we need to erase intrinsics in the typed phase, and replace them with the
                // requisite operation

                let mut ty_args = vec![];
                for arg in args {
                    ty_args.push(self.type_expr(ctx, arg)?);
                }

                comptime::resolve_intrinsic(&name, ty_args)
            }
            Expression::MethodCall(structure, method_name, args) => {
                let structure = self.type_expr(ctx, *structure)?;
                let structure_ty = &structure.ty;

                let mut ty_args = vec![];
                ty_args.push(structure.clone());
                for arg in args {
                    ty_args.push(self.type_expr(ctx, arg)?);
                }

                let method_name = format!("{}::{}", structure_ty, method_name);
                let fn_type = match ctx.get_user_def_type(&method_name) {
                    None => {
                        return Err(TypeError::new(
                            TypeErrorKind::UndefinedFunction(method_name),
                            expr.1,
                        ));
                    }
                    Some(t) => t,
                };

                let ret_ty = if let Type::Fn(fn_) = fn_type {
                    let FunctionSignature {
                        param_types,
                        ret_type,
                    } = *fn_;

                    if param_types.len() != ty_args.len() {
                        return Err(TypeError::new(
                            TypeErrorKind::FunctionArityMismatch {
                                expected: param_types.len() as u32,
                                found: ty_args.len() as u32,
                            },
                            expr.1,
                        ));
                    }

                    for (arg, ty_arg) in param_types.iter().zip(ty_args.iter()) {
                        if arg != &ty_arg.ty {
                            return Err(TypeError::new(
                                TypeErrorKind::TypeMismatch {
                                    expected: arg.clone(),
                                    found: ty_arg.ty.clone(),
                                },
                                expr.1,
                            ));
                        }
                    }
                    ret_type
                } else {
                    return Err(TypeError::new(
                        TypeErrorKind::IdentUsedAsFn(method_name),
                        expr.1,
                    ));
                };

                TypedExpression {
                    ty: ret_ty,
                    kind: ExpressionKind::FnCall(method_name, ty_args),
                    span: expr.1,
                }
            }
            Expression::Idx(arr, idx) => {
                let arr = self.type_expr(ctx, *arr)?;
                let expression_ty = match &arr.ty {
                    Type::Array(ty) => *ty.clone(),
                    _ => {
                        let err = TypeError::new(
                            TypeErrorKind::TypeMismatch {
                                expected: Type::Array(Box::new(Type::Generic("a".to_string()))),
                                found: arr.ty.clone(),
                            },
                            expr.1,
                        )
                        .add_note(format!("This has type {}", arr.ty), arr.span);
                        return Err(err);
                    }
                };
                let idx = self.type_expr(ctx, *idx)?;
                TypedExpression {
                    ty: expression_ty,
                    kind: ExpressionKind::Idx(Box::new(arr), Box::new(idx)),
                    span: expr.1,
                }
            }
            Expression::Array(elems) => {
                let mut ty_elems = vec![];
                let mut detected_type = Type::Any;
                let mut detected_span = expr.1.clone();
                for elem in elems {
                    let ty_expr = self.type_expr(ctx, elem)?;
                    if detected_type == Type::Any {
                        detected_type = ty_expr.ty.clone();
                        detected_span = ty_expr.span.clone();
                    } else if ty_expr.ty != detected_type {
                        let err = TypeError::new(
                            TypeErrorKind::TypeMismatch {
                                expected: detected_type.clone(),
                                found: ty_expr.ty.clone(),
                            },
                            expr.1.clone(),
                        )
                        .add_note(format!("This has type {}", detected_type), detected_span)
                        .add_note(format!("This has type {}", ty_expr.ty), ty_expr.span);
                        return Err(err);
                    }
                    ty_elems.push(ty_expr);
                }
                TypedExpression {
                    ty: Type::Array(Box::new(ty_elems[0].ty.clone())),
                    kind: ExpressionKind::Array(ty_elems),
                    span: expr.1,
                }
            }
            Expression::RecordInit(record_name, assignments) => {
                let mut ty_assignments = vec![];
                for assignment in assignments {
                    let (name, expr) = assignment;
                    let ty_expr = self.type_expr(ctx, expr)?;
                    ty_assignments.push((name, ty_expr));
                }

                let record_ty = ctx.get_user_def_type(&record_name);
                match record_ty {
                    Some(ty) => TypedExpression {
                        ty: ty.clone(),
                        kind: ExpressionKind::RecordInit(record_name, ty_assignments),
                        span: expr.1,
                    },
                    None => {
                        return Err(TypeError::new(
                            TypeErrorKind::UndefinedType(record_name),
                            expr.1,
                        ));
                    }
                }
            }
            Expression::Literal(lit) => TypedExpression {
                ty: match lit {
                    Literal::Int(_) => Type::Int,
                    Literal::Bool(_) => Type::Bool,
                    Literal::Str(_) => Type::Str,
                    Literal::Usize(_) => Type::Usize,
                    // Literal::Byte(_) => Type::Byte,
                },
                kind: ExpressionKind::Literal(lit),
                span: expr.1,
            },
            Expression::Ident(ident) => TypedExpression {
                ty: match ctx.var_ty(&ident) {
                    Some(ty) => ty.clone(),
                    None => {
                        return Err(TypeError::new(
                            TypeErrorKind::UndefinedVariable(ident),
                            expr.1,
                        ));
                    }
                },
                kind: ExpressionKind::Ident(ident),
                span: expr.1,
            },
            Expression::BinOp(op, lhs, rhs) => {
                let ty_lhs = self.type_expr(ctx, *lhs)?;
                let ty_rhs = self.type_expr(ctx, *rhs)?;
                let bop_ty = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => Type::Int,
                    BinOp::And
                    | BinOp::Or
                    | BinOp::EqEq
                    | BinOp::NEq
                    | BinOp::Gt
                    | BinOp::GtEq
                    | BinOp::Lt
                    | BinOp::LtEq => Type::Bool,
                };

                if ty_lhs.ty != ty_rhs.ty {
                    return Err(TypeError::new(
                        TypeErrorKind::IncompatibleTypes {
                            operation: op,
                            lhs: ty_lhs.ty,
                            rhs: ty_rhs.ty,
                        },
                        expr.1,
                    ));
                }

                if bop_ty == Type::Int && ty_lhs.ty != Type::Int {
                    return Err(TypeError::new(
                        TypeErrorKind::TypeMismatch {
                            expected: Type::Int,
                            found: ty_lhs.ty,
                        },
                        expr.1,
                    ));
                }

                TypedExpression {
                    kind: ExpressionKind::BinOp(op, Box::new(ty_lhs), Box::new(ty_rhs)),
                    ty: bop_ty,
                    span: expr.1,
                }
            }
            Expression::FnCall(name, args) => {
                let mut ty_args = vec![];
                for arg in args {
                    ty_args.push(self.type_expr(ctx, arg)?);
                }

                let name = if let Some((comptime_fn, span)) = self.comptime_fns.get(&name) {
                    let decl_ctx = unsafe { &mut *self.top_ctx };
                    let mut instantiated_comptime_types = HashMap::new();

                    #[allow(clippy::needless_range_loop)]
                    for i in 0..comptime_fn.num_comptime_args {
                        let arg_name = comptime_fn.arguments[i].0.name.clone();
                        let ExpressionKind::Type(arg_type) = &ty_args[i].kind else {
                            panic!("comptime fn arg is not a type")
                        };
                        instantiated_comptime_types.insert(arg_name, arg_type.clone());
                    }

                    println!("here: {:?}", instantiated_comptime_types);
                    let res = self.type_fn(
                        decl_ctx,
                        comptime_fn.clone(),
                        span.clone(),
                        None,
                        instantiated_comptime_types,
                    )?;
                    res.name().unwrap().to_string()
                } else {
                    name
                };

                let fn_type = match ctx.get_user_def_type(&name) {
                    None => {
                        return Err(TypeError::new(
                            TypeErrorKind::UndefinedFunction(name),
                            expr.1,
                        ));
                    }
                    Some(t) => t,
                };

                println!("2");

                let ret_ty = if let Type::Fn(fn_) = fn_type {
                    let FunctionSignature {
                        param_types,
                        ret_type,
                    } = *fn_;

                    if param_types.len() != ty_args.len() {
                        return Err(TypeError::new(
                            TypeErrorKind::FunctionArityMismatch {
                                expected: param_types.len() as u32,
                                found: ty_args.len() as u32,
                            },
                            expr.1,
                        ));
                    }

                    for (arg, ty_arg) in param_types.iter().zip(ty_args.iter()) {
                        if arg != &ty_arg.ty {
                            return Err(TypeError::new(
                                TypeErrorKind::TypeMismatch {
                                    expected: arg.clone(),
                                    found: ty_arg.ty.clone(),
                                },
                                expr.1,
                            ));
                        }
                    }
                    ret_type
                } else {
                    return Err(TypeError::new(TypeErrorKind::IdentUsedAsFn(name), expr.1));
                };

                TypedExpression {
                    kind: ExpressionKind::FnCall(name, ty_args),
                    ty: ret_ty,
                    span: expr.1,
                }
            }
            Expression::FieldAccess(expr, field_name) => {
                let expr_span = expr.1.clone();
                let ty_expr = self.type_expr(ctx, *expr)?;
                let ty = if let Type::UserDef(name) = &ty_expr.ty {
                    &ctx.get_user_def_type(name)
                        .expect("userdef types should have been registered")
                } else {
                    &ty_expr.ty
                };
                let field_ty = if let Type::Record(fields) = ty {
                    fields.iter().find_map(|(var, ty)| {
                        if *var == field_name {
                            Some(ty.clone())
                        } else {
                            None
                        }
                    })
                } else {
                    None
                };
                if field_ty.is_none() {
                    let err = TypeError::new(
                        TypeErrorKind::InvalidType(ty_expr.ty.clone()),
                        expr_span.clone(),
                    )
                    .add_note(
                        format!("Cannot access field {field_name} on type {}", ty_expr.ty),
                        expr_span,
                    );
                    return Err(err);
                }
                TypedExpression {
                    ty: field_ty.unwrap(),
                    kind: ExpressionKind::FieldAccess(Box::new(ty_expr), field_name),
                    span: expr_span,
                }
            }
            Expression::DataConstruction(construct_name, data) => {
                let (_dacon_ty, enum_ty) = match ctx.get_user_def_type(&construct_name) {
                    Some(Type::DataConstructor(dacon, data, ty_name)) => (
                        Type::DataConstructor(dacon, data, ty_name.clone()),
                        Type::UserDef(ty_name),
                    ),
                    _ => {
                        return Err(TypeError::new(
                            TypeErrorKind::UndefinedType(construct_name),
                            expr.1,
                        ));
                    }
                };

                let data = if let Some(data) = data {
                    Some(Box::new(self.type_expr(ctx, *data)?))
                } else {
                    None
                };

                TypedExpression {
                    kind: ExpressionKind::DataConstructor(construct_name, data),
                    ty: enum_ty,
                    span: expr.1,
                }
            }
        };

        Ok(expr)
    }
}
