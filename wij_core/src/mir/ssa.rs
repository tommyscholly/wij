use std::collections::{HashMap, HashSet};

use crate::{
    SizeOf,
    ast::{BinOp, Literal, Type, typed::*},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MIRType {
    Byte,
    Int,
    Usize,
    Bool,
    Str,
    Unit,
    Array(Box<MIRType>),
    Record(Vec<(String, MIRType)>),
    Fn(Vec<MIRType>, Box<MIRType>),
    Ptr,
}

impl SizeOf for MIRType {
    fn size_of(&self) -> usize {
        match self {
            MIRType::Byte => 1,
            MIRType::Int => 4,
            MIRType::Bool => 1,
            MIRType::Unit => 0,
            MIRType::Array(elem_ty) => elem_ty.size_of(),
            MIRType::Record(fields) => fields.iter().map(|(_, ty)| ty.size_of()).sum(),
            MIRType::Fn(args, ret_ty) => {
                args.iter().map(|ty| ty.size_of()).sum::<usize>() + ret_ty.size_of()
            }
            MIRType::Str => 0,
            MIRType::Ptr => 8,
            MIRType::Usize => 8,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ValueID(pub u32);
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct BlockID(pub u32);
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct FnID(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value {
    pub id: ValueID,
    pub ty: MIRType,
}

#[derive(Debug, Clone)]
pub enum Operation {
    IntConst(i32),
    UsizeConst(usize),
    BoolConst(bool),
    // unsure if i will actually be doing str constants
    // or if we need to basically allocate a str struct that has procedures on it
    StrConst(String),

    Parameter(usize), // parameter idx

    Alloca(MIRType),         // stack allocation
    Load(ValueID, MIRType),  // (address)
    Store(ValueID, ValueID), // (address, value)

    // Cranelift automatically converts to SSA, so we can break SSA here
    Assign(ValueID, ValueID),

    // array/record operations, very similar to LLVM
    GetElementPtr(ValueID, Vec<i32>), // base + indices
    #[allow(unused)]
    ExtractValue(ValueID, usize, MIRType), // for records
    #[allow(unused)]
    InsertValue(ValueID, ValueID, usize), // for records (base, new_value, field_idx)

    BinOp {
        op: BinOpKind,
        lhs: ValueID,
        rhs: ValueID,
    },

    Call {
        function: FnID,
        args: Vec<ValueID>,
    },

    Phi {
        incoming: Vec<(BlockID, ValueID)>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl From<&BinOp> for BinOpKind {
    fn from(op: &BinOp) -> Self {
        match op {
            BinOp::Add => BinOpKind::Add,
            BinOp::Sub => BinOpKind::Sub,
            BinOp::Mul => BinOpKind::Mul,
            BinOp::Div => BinOpKind::Div,
            BinOp::Mod => BinOpKind::Mod,
            BinOp::And => BinOpKind::And,
            BinOp::Or => BinOpKind::Or,
            BinOp::EqEq => BinOpKind::Eq,
            BinOp::NEq => BinOpKind::Ne,
            BinOp::Gt => BinOpKind::Gt,
            BinOp::GtEq => BinOpKind::Ge,
            BinOp::Lt => BinOpKind::Lt,
            BinOp::LtEq => BinOpKind::Le,
        }
    }
}

// terminator instruction for a basic block
#[derive(Debug, Clone)]
pub enum Terminator {
    Return(Option<ValueID>),
    // goto, but called br in llvm naming style
    Branch(BlockID),
    CondBranch {
        condition: ValueID,
        true_block: BlockID,
        false_block: BlockID,
    },
    #[allow(unused)]
    Switch {
        value: ValueID,
        cases: Vec<(i32, BlockID)>,
        default: BlockID,
    },
    #[allow(unused)]
    Call {
        function: FnID,
        args: Vec<ValueID>,
        return_to: BlockID,
    },
    Unreachable,
}

#[derive(Debug)]
pub struct Block {
    pub id: BlockID,
    pub instructions: Vec<(ValueID, Operation)>,
    pub terminator: Terminator,
    // unsure if i'm doing analysis, but if i am, this will be needed
    #[allow(unused)]
    pub(crate) predecessors: HashSet<BlockID>,
}

#[derive(Debug)]
pub struct Function {
    pub id: FnID,
    pub name: String,
    pub params: Vec<(String, MIRType)>,

    pub return_type: Option<MIRType>,

    pub entry_block: BlockID,
    pub blocks: HashMap<BlockID, Block>,

    // unsure if i'm doing analysis, but dominator tree is needed for most cfg analysis
    #[allow(unused)]
    pub(crate) dominators: HashMap<BlockID, HashSet<BlockID>>,

    // map var names to value ids
    pub symbols: HashMap<String, Value>,
}

#[derive(Debug, Default)]
pub struct Program {
    pub name: String,
    pub functions: HashMap<FnID, Function>,
    // allows faster fn lookup, but more importantly, recursive functions
    pub fns_by_name: HashMap<String, FnID>,

    pub entry_function: Option<FnID>,

    pub types: HashMap<String, MIRType>,
    pub externals: HashMap<String, (FnID, FunctionType)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionType {
    pub params: Vec<MIRType>,
    pub return_type: Option<MIRType>,
}

#[derive(Default, Debug)]
struct SSABuilder {
    next_value_id: u32,
    next_block_id: u32,
    next_fn_id: u32,

    current_function: Option<FnID>,
    current_block: Option<BlockID>,

    var_defs: HashMap<String, Value>,

    #[allow(unused)]
    incomplete_phis: Vec<(ValueID, String)>,

    entry_fn_name: Option<String>,
}

impl SSABuilder {
    fn new(entry_fn_name: Option<String>) -> Self {
        Self {
            entry_fn_name,
            ..Default::default()
        }
    }

    fn build_module(&mut self, typed_module: Module, program: &mut Program) {
        self.register_types(program, &typed_module);

        for decl in &typed_module.decls {
            match &decl.kind {
                DeclKind::Function {
                    name,
                    arguments,
                    body,
                    ret_type,
                } => {
                    println!("lowering fn: {}", name);
                    let fn_id =
                        self.lower_function(program, name, arguments, body, ret_type.clone());

                    // todo: fix this to look for main as the entry function
                    if program.entry_function.is_none() {
                        match &self.entry_fn_name {
                            Some(fnname) => {
                                if fnname == name {
                                    program.entry_function = Some(fn_id);
                                }
                            }
                            None => {
                                program.entry_function = Some(fn_id);
                            }
                        }
                    }
                }
                DeclKind::ForeignDeclarations(fds) => {
                    for (name, sig) in fds {
                        self.register_external(program, name, sig);
                    }
                }
                _ => {}
            }
        }
    }

    fn lower_function(
        &mut self,
        program: &mut Program,
        name: &str,
        arguments: &[TypedVar],
        body: &TypedStatement,
        ret_type: Option<Type>,
    ) -> FnID {
        let fn_id = FnID(self.next_fn_id);
        self.next_fn_id += 1;

        let entry_block = self.new_block();

        let mut function = Function {
            id: fn_id,
            name: name.to_string(),
            params: Vec::new(),
            return_type: ret_type
                .as_ref()
                .map(|ty| self.convert_type(ty, &program.types)),
            entry_block,
            blocks: HashMap::new(),
            dominators: HashMap::new(),
            symbols: HashMap::new(),
        };

        let mut entry = Block {
            id: entry_block,
            instructions: Vec::new(),
            terminator: Terminator::Unreachable, // filled in later
            predecessors: HashSet::new(),
        };

        self.current_function = Some(fn_id);
        self.current_block = Some(entry_block);
        self.var_defs.clear();

        for (idx, arg) in arguments.iter().enumerate() {
            let param_ty = self.convert_type(&arg.ty, &program.types);
            function.params.push((arg.id.clone(), param_ty.clone()));

            let param_value = self.new_value(param_ty);
            entry
                .instructions
                .push((param_value.id, Operation::Parameter(idx)));

            function.symbols.insert(arg.id.clone(), param_value.clone());
            self.var_defs.insert(arg.id.clone(), param_value);
        }

        program.fns_by_name.insert(name.to_string(), fn_id);
        function.blocks.insert(entry_block, entry);
        self.create_new_block(&mut function);
        self.lower_statement(program, &mut function, body);

        // if a block has no terminator, it is a return that has no value
        if let Some(block_id) = self.current_block {
            if let Some(block) = function.blocks.get_mut(&block_id) {
                if matches!(block.terminator, Terminator::Unreachable) {
                    block.terminator = Terminator::Return(None);
                }
            }
        }

        // all phis should be resolvable by the end of a function
        self.resolve_phis(&mut function);

        program.functions.insert(fn_id, function);
        fn_id
    }

    fn create_new_block(&mut self, function: &mut Function) -> BlockID {
        let block_id = self.new_block();
        if let Some(curr_block_id) = self.current_block {
            if let Some(block) = function.blocks.get_mut(&curr_block_id) {
                if matches!(block.terminator, Terminator::Unreachable) {
                    block.terminator = Terminator::Branch(block_id);
                }
            }
        }

        let block = Block {
            id: block_id,
            instructions: Vec::new(),
            terminator: Terminator::Unreachable,
            predecessors: HashSet::from_iter(vec![self.current_block.unwrap()]),
        };
        function.blocks.insert(block_id, block);
        self.current_block = Some(block_id);

        block_id
    }

    fn lower_statement(
        &mut self,
        program: &Program,
        function: &mut Function,
        stmt: &TypedStatement,
    ) {
        match &stmt.kind {
            StatementKind::Let { var, value } => {
                if let Some(init_expr) = value {
                    let init_rvalue = self.lower_expression_rval(program, function, init_expr);
                    self.var_defs.insert(var.id.clone(), init_rvalue.clone());
                    function.symbols.insert(var.id.clone(), init_rvalue);
                } else {
                    // not handling unassigned variables
                    todo!()
                }
            }
            StatementKind::Assignment(lhs, rhs) => {
                // lhs must be lval
                let lhs_value = self.lower_expression_lval(program, function, lhs);
                // rhs must be rval
                let rhs_value = self.lower_expression_rval(program, function, rhs);

                self.add_instruction_to_current_block(
                    function,
                    Operation::Store(lhs_value.id, rhs_value.id),
                    MIRType::Unit,
                );
            }

            StatementKind::Block(statements) => {
                for sub_stmt in statements {
                    self.lower_statement(program, function, sub_stmt);
                }
            }

            StatementKind::Return(expr_opt) => {
                let value = expr_opt
                    .as_ref()
                    .map(|expr| self.lower_expression_rval(program, function, expr));

                if let Some(block_id) = self.current_block {
                    if let Some(block) = function.blocks.get_mut(&block_id) {
                        block.terminator = Terminator::Return(value.map(|v| v.id));
                    }
                }

                self.current_block = Some(self.new_block());
            }

            StatementKind::If {
                condition,
                then_block,
                else_block,
            } => {
                let cond_value = self.lower_expression_rval(program, function, condition);

                let then_block_id = self.new_block();
                let else_block_id = self.new_block();
                let merge_block_id = self.new_block();

                let current_block_id = self.current_block.unwrap();
                if let Some(block) = function.blocks.get_mut(&current_block_id) {
                    block.terminator = Terminator::CondBranch {
                        condition: cond_value.id,
                        true_block: then_block_id,
                        false_block: else_block_id,
                    };
                }

                function.blocks.insert(
                    then_block_id,
                    Block {
                        id: then_block_id,
                        instructions: Vec::new(),
                        terminator: Terminator::Branch(merge_block_id),
                        predecessors: [current_block_id].into_iter().collect(),
                    },
                );

                function.blocks.insert(
                    else_block_id,
                    Block {
                        id: else_block_id,
                        instructions: Vec::new(),
                        terminator: Terminator::Branch(merge_block_id),
                        predecessors: [current_block_id].into_iter().collect(),
                    },
                );

                function.blocks.insert(
                    merge_block_id,
                    Block {
                        id: merge_block_id,
                        instructions: Vec::new(),
                        terminator: Terminator::Unreachable,
                        predecessors: [then_block_id, else_block_id].into_iter().collect(),
                    },
                );

                let outer_vars = self.var_defs.clone();

                self.current_block = Some(then_block_id);
                self.lower_statement(program, function, then_block);
                let then_vars = self.var_defs.clone();

                self.current_block = Some(else_block_id);
                if let Some(else_stmt) = else_block {
                    self.lower_statement(program, function, else_stmt);
                }
                let else_vars = self.var_defs.clone();

                self.current_block = Some(merge_block_id);

                let mut all_modified_vars = HashSet::new();
                for (var, _) in then_vars
                    .iter()
                    .filter(|(k, v)| outer_vars.get(*k) != Some(v))
                {
                    all_modified_vars.insert(var.clone());
                }

                for (var, _) in else_vars
                    .iter()
                    .filter(|(k, v)| outer_vars.get(*k) != Some(v))
                {
                    all_modified_vars.insert(var.clone());
                }

                for var in all_modified_vars {
                    let var_type = if let Some(_value_id) =
                        then_vars.get(&var).or_else(|| else_vars.get(&var))
                    {
                        // todo: fix these place holders
                        MIRType::Int // placeholder
                    } else {
                        MIRType::Int // placeholder
                    };

                    let phi_value = self.new_value(var_type.clone());

                    if let Some(block) = function.blocks.get_mut(&merge_block_id) {
                        let incoming = vec![
                            (
                                then_block_id,
                                then_vars
                                    .get(&var)
                                    .map(|v| v.id)
                                    .unwrap_or(ValueID(self.next_value_id - 1)),
                            ),
                            (
                                else_block_id,
                                else_vars
                                    .get(&var)
                                    .map(|v| v.id)
                                    .unwrap_or(ValueID(self.next_value_id - 1)),
                            ),
                        ];

                        block
                            .instructions
                            .push((phi_value.id, Operation::Phi { incoming }));
                    }

                    self.var_defs.insert(var.clone(), phi_value);
                }
            }

            StatementKind::Expression(expr) => {
                self.lower_expression_rval(program, function, expr);
            }
            _ => {
                println!("need to handle stmt: {:?}", stmt);
                todo!()
            }
        }
    }

    fn lower_expression_rval(
        &mut self,
        program: &Program,
        function: &mut Function,
        expr: &TypedExpression,
    ) -> Value {
        match &expr.kind {
            ExpressionKind::Literal(lit) => match lit {
                Literal::Int(val) => self.add_instruction_to_current_block(
                    function,
                    Operation::IntConst(*val),
                    MIRType::Int,
                ),
                Literal::Usize(val) => self.add_instruction_to_current_block(
                    function,
                    Operation::UsizeConst(*val),
                    MIRType::Usize,
                ),
                Literal::Bool(val) => self.add_instruction_to_current_block(
                    function,
                    Operation::BoolConst(*val),
                    MIRType::Bool,
                ),
                Literal::Str(val) => self.add_instruction_to_current_block(
                    function,
                    Operation::StrConst(val.clone()),
                    program.types.get("String").unwrap().clone(),
                ),
            },

            // ExpressionKind::Ident(name) => self
            //     .var_defs
            //     .get(name)
            //     .unwrap_or_else(|| {
            //         panic!("Undefined variable: {}", name);
            //     })
            //     .clone(),
            ExpressionKind::Ident(name) => {
                let value_info = self
                    .var_defs
                    .get(name)
                    .unwrap_or_else(|| panic!("Undefined variable: {}", name))
                    .clone();

                // If the var_defs stores a pointer (e.g., from Alloca),
                // and the AST type is not itself a pointer/aggregate, we need to load.
                let ast_type = &expr.ty;
                let mir_type_of_value = self.convert_type(ast_type, &program.types);

                if value_info.ty == MIRType::Ptr && mir_type_of_value != MIRType::Ptr {
                    // Check if the MIR type of the value itself is an aggregate.
                    // If `let s: MyStruct;`, `value_info` is Ptr, `mir_type_of_value` is Record.
                    // In this case, `s` used as R-value means the pointer to the struct.
                    match mir_type_of_value {
                        MIRType::Array(_) | MIRType::Record(_) => {
                            return value_info; // Pointer to aggregate is the R-value
                        }
                        _ => {
                            // It's a pointer to a scalar or simple type, load it.
                            return self.add_instruction_to_current_block(
                                function,
                                Operation::Load(value_info.id, mir_type_of_value.clone()),
                                mir_type_of_value,
                            );
                        }
                    }
                }
                value_info // Otherwise, it's already an R-value or a pointer that's used as such
            }
            ExpressionKind::FieldAccess(_, _) => {
                let field_address_value = self.lower_expression_lval(program, function, expr);

                let field_actual_mir_type = self.convert_type(&expr.ty, &program.types);
                self.add_instruction_to_current_block(
                    function,
                    Operation::Load(field_address_value.id, field_actual_mir_type.clone()),
                    field_actual_mir_type,
                )
            }
            ExpressionKind::Idx(_, _) => {
                // pass the whole expr
                let element_address_value = self.lower_expression_lval(program, function, expr);
                let element_actual_mir_type = self.convert_type(&expr.ty, &program.types);
                self.add_instruction_to_current_block(
                    function,
                    Operation::Load(element_address_value.id, element_actual_mir_type.clone()),
                    element_actual_mir_type,
                )
            }

            ExpressionKind::BinOp(op, lhs, rhs) => {
                let lhs_value = self.lower_expression_rval(program, function, lhs);
                let rhs_value = self.lower_expression_rval(program, function, rhs);
                let op_kind = BinOpKind::from(op);
                self.add_instruction_to_current_block(
                    function,
                    Operation::BinOp {
                        op: op_kind,
                        lhs: lhs_value.id,
                        rhs: rhs_value.id,
                    },
                    self.convert_type(&expr.ty, &program.types),
                )
            }

            ExpressionKind::FnCall(func_name, args) => {
                let arg_values = args
                    .iter()
                    .map(|arg| {
                        let value = self.lower_expression_rval(program, function, arg);
                        value.id
                    })
                    .collect();

                let fn_id = program
                    .fns_by_name
                    .get(func_name.as_str())
                    .unwrap_or_else(|| {
                        println!("{:?}", program.fns_by_name);
                        panic!(
                            "function {} should exist if we are trying to call it",
                            func_name.as_str(),
                        )
                    });

                self.add_instruction_to_current_block(
                    function,
                    Operation::Call {
                        function: *fn_id,
                        args: arg_values,
                    },
                    self.convert_type(&expr.ty, &program.types),
                )
            }

            ExpressionKind::Array(elements) => {
                let elem_type =
                    if let MIRType::Array(elem_ty) = self.convert_type(&expr.ty, &program.types) {
                        *elem_ty
                    } else {
                        unreachable!();
                    };

                let array_ptr = self.add_instruction_to_current_block(
                    function,
                    Operation::Alloca(MIRType::Array(Box::new(elem_type.clone()))),
                    MIRType::Ptr,
                );

                for (i, elem) in elements.iter().enumerate() {
                    let elem_value = self.lower_expression_rval(program, function, elem);

                    let elem_ptr = self.add_instruction_to_current_block(
                        function,
                        Operation::GetElementPtr(array_ptr.id, vec![i as i32]),
                        MIRType::Ptr,
                    );

                    self.add_instruction_to_current_block(
                        function,
                        Operation::Store(elem_ptr.id, elem_value.id),
                        MIRType::Unit,
                    );
                }

                array_ptr
            }

            ExpressionKind::RecordInit(record_type_name, assignments) => {
                // record lits are stack-allocated, expression evaluates to a pointer.
                let record_mir_ty = program.types.get(record_type_name).unwrap().clone();

                let record_ptr_val = self.add_instruction_to_current_block(
                    function,
                    Operation::Alloca(record_mir_ty),
                    MIRType::Ptr,
                );

                for (field_name, field_expr) in assignments {
                    let field_rvalue = self.lower_expression_rval(program, function, field_expr);
                    let field_idx = self
                        .record_field_idx(program, record_type_name, field_name)
                        .unwrap();

                    // todo: eliminate insertval and extract val
                    let field_ptr = self.add_instruction_to_current_block(
                        function,
                        Operation::GetElementPtr(record_ptr_val.id, vec![field_idx as i32]),
                        MIRType::Ptr,
                    );
                    self.add_instruction_to_current_block(
                        function,
                        Operation::Store(field_ptr.id, field_rvalue.id),
                        MIRType::Unit,
                    );
                }
                record_ptr_val
            }
            ExpressionKind::Self_ => function
                .symbols
                .get("self")
                .expect("self not found")
                .clone(),
            e => {
                println!("need to handle expr: {:#?}", e);

                todo!()
            }
        }
    }

    // should always return a Value with MIRType::Ptr
    fn lower_expression_lval(
        &mut self,
        program: &Program,
        function: &mut Function,
        expr: &TypedExpression,
    ) -> Value {
        match &expr.kind {
            ExpressionKind::Ident(name) => {
                let value_info = self
                    .var_defs
                    .get(name)
                    .unwrap_or_else(|| panic!("Undefined variable: {}", name))
                    .clone();
                if !matches!(value_info.ty, MIRType::Ptr) {
                    panic!(
                        "Identifier '{}' is not a pointer, cannot be direct L-value for Store without Alloca",
                        name
                    );
                }
                value_info
            }
            ExpressionKind::FieldAccess(record_expr, field_name) => {
                let record_ptr_value = self.lower_expression_rval(program, function, record_expr);

                // todo: remove records and replace with ptrs
                if !matches!(record_ptr_value.ty, MIRType::Ptr | MIRType::Record(_)) {
                    panic!(
                        "Base of field access must be a pointer. Got: {:?}",
                        record_ptr_value.ty
                    );
                }

                let field_idx = match &record_expr.ty {
                    Type::UserDef(record_name_ast) => self
                        .record_field_idx(program, record_name_ast, field_name)
                        .unwrap_or_else(|| {
                            panic!(
                                "Field {} not found in record {}",
                                field_name, record_name_ast
                            )
                        }) as i32,
                    Type::Record(ast_fields) => ast_fields
                        .iter()
                        .position(|(name, _)| name == field_name)
                        .unwrap_or_else(|| {
                            panic!("Field {} not found in anonymous record", field_name)
                        }) as i32,
                    _ => panic!("Field access on non-record type: {:?}", record_expr.ty),
                };

                self.add_instruction_to_current_block(
                    function,
                    Operation::GetElementPtr(record_ptr_value.id, vec![0, field_idx]),
                    MIRType::Ptr,
                )
            }
            ExpressionKind::Idx(array_expr, index_expr) => {
                let array_ptr_value = self.lower_expression_rval(program, function, array_expr);
                if !matches!(array_ptr_value.ty, MIRType::Ptr) {
                    panic!(
                        "Base of index access must be a pointer. Got: {:?}",
                        array_ptr_value.ty
                    );
                }

                let index_const = self.lower_expression_rval(program, function, index_expr);
                let idx_val_for_gep = match function
                    .blocks
                    .get(&self.current_block.unwrap())
                    .unwrap()
                    .instructions
                    .iter()
                    .find(|(id, _)| *id == index_const.id)
                {
                    Some((_, Operation::IntConst(i))) => *i,
                    Some((_, Operation::UsizeConst(u))) => *u as i32,
                    _ => panic!(
                        "Dynamic GEP index not yet fully supported in this example, expected const int/usize"
                    ),
                };

                self.add_instruction_to_current_block(
                    function,
                    Operation::GetElementPtr(array_ptr_value.id, vec![idx_val_for_gep]),
                    MIRType::Ptr,
                )
            }
            _ => panic!(
                "Expression kind {:?} cannot be an L-value to get an address from",
                expr.kind
            ),
        }
    }

    fn add_instruction_to_current_block(
        &mut self,
        function: &mut Function,
        operation: Operation,
        ty: MIRType,
    ) -> Value {
        let value = self.new_value(ty);
        let value_id = value.id;

        if let Some(block_id) = self.current_block {
            if let Some(block) = function.blocks.get_mut(&block_id) {
                block.instructions.push((value_id, operation));
            } else {
                let block = Block {
                    id: block_id,
                    instructions: vec![(value_id, operation)],
                    terminator: Terminator::Unreachable, // tmp, filled in above
                    predecessors: HashSet::new(),
                };
                function.blocks.insert(block_id, block);
            }
        }

        value
    }

    fn register_external(&mut self, program: &mut Program, name: &str, sig: &FunctionSignature) {
        let param_types = sig
            .param_types
            .iter()
            .map(|ty| self.convert_type(ty, &program.types))
            .collect();
        let return_type = self.convert_type(&sig.ret_type, &program.types);
        // believe externals still need fn_ids
        let external_fn_id = FnID(self.next_fn_id);
        self.next_fn_id += 1;

        program.externals.insert(
            name.to_string(),
            (
                external_fn_id,
                FunctionType {
                    params: param_types,
                    return_type: Some(return_type),
                },
            ),
        );

        program.fns_by_name.insert(name.to_string(), external_fn_id);
    }

    fn record_field_idx(
        &self,
        program: &Program,
        record_name: &str,
        field_name: &str,
    ) -> Option<usize> {
        match program.types.get(record_name) {
            Some(MIRType::Record(fields)) => fields.iter().position(|(name, _)| name == field_name),
            _ => None,
        }
    }

    fn register_types(&mut self, program: &mut Program, typed_module: &Module) {
        program.types.insert(
            "String".to_string(),
            MIRType::Record(vec![
                ("buffer".to_string(), MIRType::Ptr),
                ("length".to_string(), MIRType::Int),
            ]),
        );

        for decl in &typed_module.decls {
            match &decl.kind {
                DeclKind::Record { name, fields } => {
                    let mir_fields = fields
                        .iter()
                        .map(|(field_name, ty)| {
                            (field_name.clone(), self.convert_type(ty, &program.types))
                        })
                        .collect();

                    program
                        .types
                        .insert(name.clone(), MIRType::Record(mir_fields));
                }
                DeclKind::Enum {
                    name: _,
                    variants: _,
                } => {
                    // need to figure out how I am representing enums
                    todo!()
                }
                _ => {}
            }
        }
    }

    fn resolve_phis(&mut self, _function: &mut Function) {
        // todo
        #[allow(clippy::needless_return)]
        return;
    }

    fn new_value(&mut self, ty: MIRType) -> Value {
        let id = ValueID(self.next_value_id);
        self.next_value_id += 1;
        Value { id, ty }
    }

    fn new_block(&mut self) -> BlockID {
        let id = BlockID(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    #[allow(clippy::only_used_in_recursion)]
    fn convert_type(&self, ty: &Type, types: &HashMap<String, MIRType>) -> MIRType {
        match ty {
            Type::Byte => MIRType::Byte,
            Type::Int => MIRType::Int,
            Type::Usize => MIRType::Usize,
            Type::Bool => MIRType::Bool,
            Type::Str => MIRType::Str,
            Type::Unit => MIRType::Unit,
            Type::Array(elem_ty) => MIRType::Array(Box::new(self.convert_type(elem_ty, types))),
            Type::Record(fields) => MIRType::Record(
                fields
                    .iter()
                    .map(|(name, ty)| (name.clone(), self.convert_type(ty, types)))
                    .collect(),
            ),
            Type::Fn(sig) => {
                let param_types = sig
                    .param_types
                    .iter()
                    .map(|ty| self.convert_type(ty, types))
                    .collect();
                MIRType::Fn(
                    param_types,
                    Box::new(self.convert_type(&sig.ret_type, types)),
                )
            }
            Type::UserDef(name) => {
                if let Some(ty) = types.get(name) {
                    ty.clone()
                } else {
                    panic!("unbound type {}", name)
                }
            }
            Type::Ptr(_) => MIRType::Ptr,
            Type::OpaquePtr => MIRType::Ptr,
            t => panic!("unhandled type: {}", t), // todo: handle other types
        }
    }
}

pub fn build_ssa(mut modules: Vec<Module>) -> Program {
    let mut program = Program {
        name: "main".to_string(),
        functions: HashMap::new(),
        fns_by_name: HashMap::new(),
        entry_function: None,
        types: HashMap::new(),
        externals: HashMap::new(),
    };

    let mut ssa_builder = SSABuilder::new(Some("main".to_string()));
    while let Some(typed_module) = modules.pop() {
        ssa_builder.build_module(typed_module, &mut program);
    }

    program
}
