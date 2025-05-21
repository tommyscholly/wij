// This MIR is a CFG
// From the MIR, we then lower into our target IR depending on the codegen backend
// Once we lower to MIR, any errors here are ICE
#![allow(dead_code)]
#![allow(unused_variables)]

pub mod ssa;

// use std::collections::HashMap;
//
// use crate::{
//     Module,
//     ast::{
//         Literal, Type,
//         typed::{
//             DeclKind, ExpressionKind, FunctionSignature, StatementKind, TyCtx, TypedExpression,
//             TypedStatement, TypedVar,
//         },
//     },
// };
//
// #[derive(Debug, Clone, PartialEq, Eq)]
// enum MIRType {
//     Int,
//     Bool,
//     Record(Vec<MIRType>),
//     Fn(Vec<MIRType>, Box<MIRType>),
// }
//
// impl From<Type> for MIRType {
//     fn from(value: Type) -> Self {
//         match value {
//             Type::Int => MIRType::Int,
//             Type::Bool => MIRType::Bool,
//             Type::Fn(sig) => {
//                 let args = sig
//                     .param_types
//                     .iter()
//                     .map(|ty| MIRType::from(ty.clone()))
//                     .collect();
//                 let ret = Box::new(MIRType::from(sig.ret_type.clone()));
//                 MIRType::Fn(args, ret)
//             }
//             _ => todo!(),
//         }
//     }
// }
//
// #[derive(Clone, Copy, PartialEq, Eq, Debug)]
// struct BlockID(u32);
// #[derive(Clone, Copy, PartialEq, Eq, Debug)]
// struct LocalID(u32);
// #[derive(Clone, Copy, PartialEq, Eq, Debug)]
// struct ParamID(u32);
// #[derive(Clone, Copy, PartialEq, Eq, Debug)]
// struct FnID(u32);
//
// #[derive(Debug, Default)]
// struct LabelBuilder {
//     block_id_ctr: u32,
//     local_id_ctr: u32,
//     param_id_ctr: u32,
//     fn_id_ctr: u32,
// }
//
// impl LabelBuilder {
//     fn new() -> Self {
//         Self::default()
//     }
//
//     fn block_id(&mut self) -> BlockID {
//         let id = self.block_id_ctr;
//         self.block_id_ctr += 1;
//         BlockID(id)
//     }
//
//     fn local_id(&mut self) -> LocalID {
//         let id = self.local_id_ctr;
//         self.local_id_ctr += 1;
//         LocalID(id)
//     }
//
//     fn param_id(&mut self) -> ParamID {
//         let id = self.param_id_ctr;
//         self.param_id_ctr += 1;
//         ParamID(id)
//     }
//
//     fn fn_id(&mut self) -> FnID {
//         let id = self.fn_id_ctr;
//         self.fn_id_ctr += 1;
//         FnID(id)
//     }
// }
//
// #[derive(Debug, Clone)]
// enum Data {
//     Local(LocalID, MIRType),
//     Param(ParamID, MIRType),
//     // index into an offset at data 0 by data 1
//     Offset(Box<Data>, Box<Data>),
// }
//
// #[derive(Debug)]
// enum Trans {
//     Goto {
//         target: BlockID,
//     },
//     Call {
//         target: FnID,
//         args: Vec<Data>,
//         return_point: Option<BlockID>,
//     },
//     Return {
//         value: Option<Data>,
//     },
//     // When codegening, we will always emit a joining block, so the false_block will point to the
//     // joining block if there is no actual false block
//     If {
//         cond: Data,
//         true_block: BlockID,
//         false_block: BlockID,
//     },
// }
//
// #[derive(Debug)]
// enum Value {
//     Immediate(i32),
//     Bool(bool),
//     Data(Data),
//     // Array loading can also be used for records
//     Array(Vec<Value>),
// }
//
// #[derive(Debug)]
// enum Statement {
//     // We need to have some sort of escape analysis to demote to a heap allocation
//     // Register (1)
//     // Stack (2)
//     // Heap (3)
//     Allocate { local: LocalID, ty: MIRType },
//     Store { location: Data, value: Value },
//     Move { src: Data, dst: Data },
//     // Loops should drop into a block, which will end in a Trans::Goto to the start
//     Loop(BlockID),
// }
//
// impl Statement {
//     fn lower(cfg: &mut Cfg, stmt: TypedStatement) -> Vec<Self> {
//         match stmt.kind {
//             StatementKind::Let { var, value } => {
//                 let local_id = cfg.label_builder.local_id();
//                 match value {
//                     // None => vec![Statement::Allocate {
//                     //     local: local_id,
//                     //     ty: MIRType::from(var.ty),
//                     // }],
//
//                     // should we emit anything?
//                     None => vec![],
//                     Some(value) => {
//                         let mir_ty = MIRType::from(var.ty);
//                         let mut expr = Statement::lower_expr(cfg, value);
//
//                         let data = Data::Local(local_id, mir_ty);
//                         cfg.var_data_map.insert(var.id, data.clone());
//
//                         let store = Statement::Store {
//                             location: data,
//                             value: Value::Data(expr.last().unwrap().to_data()),
//                         };
//
//                         expr.push(store);
//                         expr
//                     }
//                 }
//             }
//             _ => todo!(),
//         }
//     }
//
//     fn lower_expr(cfg: &mut Cfg, expr: TypedExpression) -> Vec<Self> {
//         match expr.kind {
//             ExpressionKind::Literal(lit) => vec![match lit {
//                 Literal::Int(i) => Statement::Store {
//                     location: Data::Local(cfg.label_builder.local_id(), MIRType::Int),
//                     value: Value::Immediate(i),
//                 },
//                 Literal::Bool(b) => Statement::Store {
//                     location: Data::Local(cfg.label_builder.local_id(), MIRType::Bool),
//                     value: Value::Bool(b),
//                 },
//                 _ => todo!(),
//             }],
//             ExpressionKind::Ident(var) => {
//                 // todo: avoid this load
//                 let data_loc = cfg.var_data_map.get(&var).unwrap().clone();
//                 let tmp_local_id = cfg.label_builder.local_id();
//                 // safety: all variables should have types by the end of type checking
//                 let var_ty = cfg.tyctx.get_user_def_type(&var).unwrap();
//                 vec![Statement::Move {
//                     src: data_loc,
//                     dst: Data::Local(tmp_local_id, MIRType::from(var_ty)),
//                 }]
//             }
//             ExpressionKind::Array(exprs) => {
//                 let mut stmts: Vec<Self> = exprs
//                     .into_iter()
//                     .map(|expr| Statement::lower_expr(cfg, expr))
//                     .flatten()
//                     .collect();
//
//                 let values = stmts
//                     .iter()
//                     .map(|stmt| Value::Data(stmt.to_data()))
//                     .collect();
//
//                 stmts.push(Statement::Store {
//                     location: Data::Local(cfg.label_builder.local_id(), MIRType::Int),
//                     value: Value::Array(values),
//                 });
//
//                 stmts
//             }
//             ExpressionKind::Idx(expr, idx_expr) => {
//                 let mut to_idx_expr = Statement::lower_expr(cfg, *expr);
//                 let to_idx_expr_data_loc = to_idx_expr.last().unwrap().to_data();
//
//                 let mut idx_expr = Statement::lower_expr(cfg, *idx_expr);
//                 let idx_expr_data_loc = idx_expr.last().unwrap().to_data();
//                 let tmp_idx_local_id = cfg.label_builder.local_id();
//                 let stmt = Statement::Move {
//                     src: Data::Offset(Box::new(to_idx_expr_data_loc), Box::new(idx_expr_data_loc)),
//                     dst: Data::Local(tmp_idx_local_id, MIRType::Int),
//                 };
//
//                 to_idx_expr.append(&mut idx_expr);
//                 to_idx_expr.push(stmt);
//                 to_idx_expr
//             }
//         }
//     }
//
//     fn to_data(&self) -> Data {
//         match self {
//             Statement::Store { location, .. } => location.clone(),
//             Statement::Move { dst, .. } => dst.clone(),
//             Statement::Allocate { local, ty } => Data::Local(*local, ty.clone()),
//             Statement::Loop(_) => panic!("cannot convert loop to data"),
//         }
//     }
// }
//
// #[derive(Debug)]
// struct Block {
//     id: BlockID,
//     params: Vec<(ParamID, MIRType)>,
//     stmts: Vec<Statement>,
//     trans: Trans,
// }
//
// impl Block {
//     fn split_block_by_trans(cfg: &mut Cfg, mut stmts: Vec<TypedStatement>) -> Vec<Block> {
//         let mut curr_block_id = cfg.label_builder.block_id();
//         let mut next_block_id = cfg.label_builder.block_id();
//         let mut curr_block_params = vec![];
//         let mut next_block_params = vec![];
//         let mut blocks = vec![];
//         let mut current_stmts: Vec<Statement> = vec![];
//
//         while let Some(stmt) = stmts.pop() {
//             match stmt.kind {
//                 StatementKind::Expression(expr) => match expr.kind {
//                     ExpressionKind::FnCall(name, args) => {
//                         let mut mir_arg_stmts: Vec<Statement> = args
//                             .into_iter()
//                             .map(|arg| Statement::lower_expr(cfg, arg))
//                             .collect();
//
//                         let mir_args = mir_arg_stmts.iter().map(|arg| arg.to_data()).collect();
//                         current_stmts.append(&mut mir_arg_stmts);
//
//                         let fn_id = cfg.fn_id_map.get(&name).unwrap();
//                         let ret_point = if stmts.is_empty() {
//                             None
//                         } else {
//                             // if we're returning, we take the ret type of the function we're
//                             // calling, and store it as a parameter of the next block
//                             let fn_sig = cfg.fn_by_id(*fn_id); // immut borrow
//                             // clone is needed to avoid mutable and immutable borrows
//                             if let Some(ty) = fn_sig.ret_type.clone() {
//                                 // mut borrow in param_id()
//                                 next_block_params.push((cfg.label_builder.param_id(), ty))
//                             }
//                             Some(next_block_id)
//                         };
//
//                         let trans = Trans::Call {
//                             // ids impl copy
//                             target: *fn_id,
//                             args: mir_args,
//                             return_point: ret_point,
//                         };
//                         let block = Block {
//                             id: curr_block_id,
//                             params: curr_block_params,
//                             stmts: current_stmts,
//                             trans,
//                         };
//
//                         blocks.push(block);
//                         current_stmts = vec![];
//                         curr_block_params = next_block_params;
//                         next_block_params = vec![];
//                         curr_block_id = next_block_id;
//                         next_block_id = cfg.label_builder.block_id();
//                     }
//                     _ => current_stmts.push(Statement::lower_expr(cfg, *expr)),
//                 },
//                 StatementKind::Break => {
//                     let block = Block {
//                         id: curr_block_id,
//                         params: curr_block_params,
//                         stmts: current_stmts,
//                         trans: Trans::Goto {
//                             target: next_block_id,
//                         },
//                     };
//                     blocks.push(block);
//                     current_stmts = vec![];
//                     curr_block_params = next_block_params;
//                     next_block_params = vec![];
//                     curr_block_id = next_block_id;
//                     next_block_id = cfg.label_builder.block_id();
//                 }
//                 StatementKind::Continue => {
//                     let block = Block {
//                         id: curr_block_id,
//                         params: curr_block_params,
//                         stmts: current_stmts,
//                         trans: Trans::Goto {
//                             target: curr_block_id,
//                         },
//                     };
//                     blocks.push(block);
//                     current_stmts = vec![];
//                     curr_block_params = next_block_params;
//                     next_block_params = vec![];
//                     curr_block_id = next_block_id;
//                     next_block_id = cfg.label_builder.block_id();
//                 }
//                 StatementKind::Return(value) => {
//                     let return_stmt = value.map(|expr| Statement::lower_expr(cfg, expr));
//                     let return_data = return_stmt.as_ref().map(|stmt| stmt.to_data());
//
//                     if let Some(stmt) = return_stmt {
//                         current_stmts.push(stmt);
//                     }
//
//                     let block = Block {
//                         id: curr_block_id,
//                         params: curr_block_params,
//                         stmts: current_stmts,
//                         trans: Trans::Return { value: return_data },
//                     };
//                     blocks.push(block);
//                     current_stmts = vec![];
//                     curr_block_params = next_block_params;
//                     next_block_params = vec![];
//                     curr_block_id = next_block_id;
//                     next_block_id = cfg.label_builder.block_id();
//                 }
//                 StatementKind::If {
//                     condition,
//                     then_block,
//                     else_block,
//                 } => {
//                     let mut then_blocks = Self::lower(cfg, *then_block);
//                     // safety: guaranteed at least one then_block
//                     let then_block_start_id = then_blocks[0].id;
//                     let else_blocks = else_block.map(|else_block| Self::lower(cfg, *else_block));
//                     let else_block_start_id = if let Some(ebs) = &else_blocks {
//                         ebs[0].id
//                     } else {
//                         next_block_id
//                     };
//
//                     let cond = Statement::lower_expr(cfg, condition);
//                     let cond_data = cond.to_data();
//                     current_stmts.push(cond);
//
//                     let if_block = Block {
//                         id: curr_block_id,
//                         params: curr_block_params,
//                         stmts: current_stmts,
//                         trans: Trans::If {
//                             cond: cond_data,
//                             true_block: then_block_start_id,
//                             false_block: else_block_start_id,
//                         },
//                     };
//
//                     blocks.push(if_block);
//                     // the if statement blocks always appear after the block we were coming from
//                     blocks.append(&mut then_blocks);
//                     blocks.append(&mut else_blocks.unwrap_or_default());
//
//                     current_stmts = vec![];
//                     curr_block_params = next_block_params;
//                     next_block_params = vec![];
//                     curr_block_id = next_block_id;
//                     next_block_id = cfg.label_builder.block_id();
//                 }
//                 _ => current_stmts.extend(Statement::lower(cfg, stmt)),
//             }
//         }
//
//         blocks
//     }
//
//     fn lower(cfg: &mut Cfg, body: TypedStatement) -> Vec<Self> {
//         let StatementKind::Block(stmts) = body.kind else {
//             unreachable!()
//         };
//
//         Block::split_block_by_trans(cfg, stmts)
//     }
// }
//
// #[derive(Debug)]
// struct Function {
//     id: FnID,
//     params: Vec<(ParamID, MIRType)>,
//     entry_block: BlockID,
//     blocks: Vec<Block>,
//     ret_type: Option<MIRType>,
//     visible: bool,
// }
//
// impl Function {
//     fn lower(
//         cfg: &mut Cfg,
//         fn_id: FnID,
//         arguments: Vec<TypedVar>,
//         body: TypedStatement,
//         ret_type: Option<Type>,
//         visible: bool,
//     ) -> Self {
//         let mir_args = arguments
//             .into_iter()
//             .map(|var| {
//                 let param_id = cfg.label_builder.param_id();
//                 let mir_ty = MIRType::from(var.ty);
//
//                 (param_id, mir_ty)
//             })
//             .collect();
//
//         let blocks = Block::lower(cfg, body);
//         // safety: guaranteed at least one block, i think??
//         let entry_block_id = blocks[0].id;
//
//         // last block should always end in a return, and we can check that for safety in testing
//         #[cfg(debug_assertions)]
//         let last_block = blocks.last().unwrap();
//
//         #[cfg(debug_assertions)]
//         match &ret_type {
//             Some(_ty) => {
//                 if let Trans::Return { value } = &last_block.trans {
//                     assert!(value.is_some());
//                 } else {
//                     panic!("expected return type, got: {:?}", last_block.trans);
//                 }
//             }
//             None => {
//                 if let Trans::Return { value } = &last_block.trans {
//                     assert!(value.is_none());
//                 } else {
//                     panic!("expected return type, got: {:?}", last_block.trans);
//                 }
//             }
//         };
//
//         let ret_type = ret_type.map(MIRType::from);
//
//         Function {
//             id: fn_id,
//             params: mir_args,
//             entry_block: entry_block_id,
//             blocks,
//             ret_type,
//             visible,
//         }
//     }
// }
//
// #[derive(Default, Debug)]
// pub struct Cfg {
//     tyctx: TyCtx,
//     label_builder: LabelBuilder,
//     fn_id_map: HashMap<String, FnID>,
//     var_data_map: HashMap<String, Data>,
//     functions: Vec<Function>,
//     user_def_types: HashMap<String, MIRType>,
//     // constraint: this MIRType is guaranteed to be MIRType::Fn
//     foreign_imports: HashMap<String, MIRType>,
// }
//
// impl Cfg {
//     fn new(tyctx: TyCtx) -> Cfg {
//         Self {
//             tyctx,
//             ..Default::default()
//         }
//     }
//
//     fn fn_by_id(&self, fn_id: FnID) -> &Function {
//         self.functions.iter().find(|f| f.id == fn_id).unwrap()
//     }
//
//     fn convert_type_decl(&mut self, decl: DeclKind) {
//         match decl {
//             DeclKind::Record { name, fields } => {
//                 let mir_fields = fields
//                     .into_iter()
//                     .map(|(_, ty)| MIRType::from(ty))
//                     .collect();
//                 let mir_type = MIRType::Record(mir_fields);
//                 self.user_def_types.insert(name, mir_type);
//             }
//             DeclKind::Enum { .. } => {}
//             _ => unreachable!(),
//         }
//     }
//
//     // this should be enough information for us to emit LLVM instructions for external functions
//     fn register_foreign_imports(&mut self, fds: Vec<(String, FunctionSignature)>) {
//         for (fn_name, fd) in fds {
//             let fn_ty = Type::Fn(Box::new(fd));
//             let mir_fn_ty = MIRType::from(fn_ty);
//             self.foreign_imports.insert(fn_name, mir_fn_ty);
//         }
//     }
//
//     fn assign_fn_id(&mut self, fn_name: String) -> FnID {
//         let fn_id = self.label_builder.fn_id();
//         self.fn_id_map.insert(fn_name, fn_id);
//         fn_id
//     }
// }
//
// pub fn generate_mir(module: Module, tyctx: TyCtx) -> Cfg {
//     let mut cfg = Cfg::new(tyctx);
//
//     for decl in module.decls.into_iter() {
//         match decl.kind {
//             DeclKind::Record { .. } => cfg.convert_type_decl(decl.kind),
//             DeclKind::Enum { .. } => cfg.convert_type_decl(decl.kind),
//             DeclKind::ForeignDeclarations(ds) => cfg.register_foreign_imports(ds),
//             DeclKind::Procedures(_, _) => todo!(),
//             DeclKind::Function {
//                 name,
//                 arguments,
//                 body,
//                 ret_type,
//             } => {
//                 let fn_id = cfg.assign_fn_id(name);
//                 let mir_fn =
//                     Function::lower(&mut cfg, fn_id, arguments, body, ret_type, decl.visible);
//                 cfg.functions.push(mir_fn);
//             }
//         }
//     }
//
//     cfg
// }
