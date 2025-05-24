#![allow(unused_imports)]

use std::collections::HashMap;

use cranelift::codegen;
use cranelift::frontend;

use codegen::entity::EntityRef;
use codegen::ir::{AbiParam, Function, InstBuilder, Signature, UserFuncName, types::*};
use codegen::isa::CallConv;
use codegen::settings::{self, Flags};
use codegen::verifier::verify_function;
use cranelift_module::FuncId;
use cranelift_module::Linkage;
use cranelift_module::{Module, default_libcall_names};
use cranelift_object::{ObjectBuilder, ObjectModule};
use frontend::{FunctionBuilder, FunctionBuilderContext, Variable};

use wij_core::{
    Program, ssa::Block as SSABlock, ssa::Function as SSAFunction, ssa::MIRType,
    ssa::Operation as SSAOperation,
};

#[derive(Default, Debug)]
struct ProgramCtx {
    // maps fnid to funcid
    fnid_to_funcid: HashMap<u32, FuncId>,
    variables: HashMap<u32, Variable>,
}

impl ProgramCtx {
    fn declare_variable(
        &mut self,
        varid: u32,
        builder: &mut FunctionBuilder,
        ty: Type,
    ) -> Variable {
        let var = Variable::new(varid as usize);
        builder.declare_var(var, ty);
        self.variables.insert(varid, var);
        var
    }
}

struct CraneliftProgram {
    fbctx: FunctionBuilderContext,
    module: ObjectModule,
    pctx: ProgramCtx,
}

struct FunctionTranslator<'ctx> {
    builder: FunctionBuilder<'ctx>,
    module: &'ctx mut ObjectModule,
    pctx: &'ctx mut ProgramCtx,
}

impl CraneliftProgram {
    fn new(program_name: &str) -> Self {
        let flags = Flags::new(settings::builder());

        let isa_builder = cranelift_native::builder().expect("arch isnt supported");
        let isa = isa_builder.finish(flags).expect("isa builder not finished");

        let object_builder = ObjectBuilder::new(isa, program_name, default_libcall_names())
            .expect("object builder not supported");
        let module = ObjectModule::new(object_builder);

        let fbctx = FunctionBuilderContext::new();
        let pctx = ProgramCtx::default();
        Self {
            fbctx,
            module,
            pctx,
        }
    }

    // for want of a better word than compile
    fn compile(&mut self, program: Program) {
        for (fnid, func) in program.functions {
            self.compile_function(fnid.0, func);
        }
    }

    fn compile_function(&mut self, fnid: u32, func: SSAFunction) {
        let param_tys: Vec<Type> = func.params.iter().map(|(_, ty)| ty.to_type()).collect();
        let mut function_signature = self.module.make_signature();
        for ty in &param_tys {
            function_signature.params.push(AbiParam::new(*ty));
        }

        if let Some(ret_ty) = &func.return_type {
            function_signature
                .returns
                .push(AbiParam::new(ret_ty.to_type()));
        }

        // todo: add actual linkage determination based on pub/priv
        let linkage = Linkage::Export;
        let func_id = self
            .module
            .declare_function(&func.name, linkage, &function_signature)
            // todo: error handling
            .unwrap();

        self.pctx.fnid_to_funcid.insert(fnid, func_id);

        let mut fctx = self.module.make_context();
        fctx.func.signature = function_signature;

        let mut fb = FunctionBuilder::new(&mut fctx.func, &mut self.fbctx);

        // initial function setup
        let block = fb.create_block();
        fb.append_block_params_for_function_params(block);
        fb.switch_to_block(block);
        fb.seal_block(block);

        for (idx, (param_name, ty)) in func.params.iter().enumerate() {
            // need to get the param_id because all references to the param is the id
            let param_id = func.symbols.get(param_name).unwrap().0;
            let param_var = self.pctx.declare_variable(param_id, &mut fb, ty.to_type());
            let param_val = fb.block_params(block)[idx];
            fb.def_var(param_var, param_val);
        }

        let mut fntrans = FunctionTranslator::new(fb, &mut self.module, &mut self.pctx);
        fntrans.translate_function(func);

        if let Err(errors) = verify_function(fntrans.builder.func, fntrans.module.isa()) {
            panic!("Function verification failed:\n{}", errors);
        }

        println!("{}", fntrans.builder.func.display());

        fntrans.builder.finalize();
        self.module.define_function(func_id, &mut fctx).unwrap();
    }
}

impl<'ctx> FunctionTranslator<'ctx> {
    fn new(
        builder: FunctionBuilder<'ctx>,
        module: &'ctx mut ObjectModule,
        pctx: &'ctx mut ProgramCtx,
    ) -> Self {
        Self {
            builder,
            module,
            pctx,
        }
    }

    fn translate_function(&mut self, mut func: SSAFunction) {
        let entry_block = func.blocks.remove(&func.entry_block).unwrap();
    }

    fn translate_block(&mut self, block: SSABlock) {
        for (val_id, operation) in block.instructions {
            self.translate_operation(val_id.0, operation);
        }
    }

    fn translate_operation(&mut self, val_id: u32, oper: SSAOperation) {
        use SSAOperation::*;
        match oper {
            IntConst(i) => {
                let ty = I32;
                let const_val = self.builder.ins().iconst(ty, i as i64);
                let var = self.pctx.declare_variable(val_id, &mut self.builder, ty);
                self.builder.def_var(var, const_val);
            }
            BoolConst(b) => {
                let val = if b { 1 } else { 0 };
                let ty = I8;
                let const_val = self.builder.ins().iconst(ty, val);
                let var = self.pctx.declare_variable(val_id, &mut self.builder, ty);
                self.builder.def_var(var, const_val);
            }
            _ => todo!(),
        }
    }
}

trait ToType {
    fn to_type(&self) -> Type;
}

impl ToType for MIRType {
    fn to_type(&self) -> Type {
        match self {
            MIRType::Byte => I8,
            MIRType::Int => I32,
            MIRType::Bool => I8,
            MIRType::Unit => I8,
            // all i64s are ptrs
            MIRType::Array(_elem_ty) => I64,
            MIRType::Record(_fields) => I64,
            MIRType::Fn(_args, _ret_ty) => I64,
            MIRType::Str => I64,
            MIRType::Ptr => I64,
        }
    }
}

#[allow(unused)]
pub fn compile(program: Program, program_name: &str) {
    let cranelift_prog = CraneliftProgram::new(program_name);
}
