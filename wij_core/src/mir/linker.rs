#![allow(dead_code, unused)]

// This is unfinished
// My current linking strategy is link every module separately AFTER codegen
// A better solution would be to link every module we have so far, then do codegen

use crate::Module;
use crate::Program;
use crate::WijError;
use crate::ast::Type;
use crate::ast::typed::DeclKind;
use crate::ast::typed::ExpressionKind;
use crate::ast::typed::FunctionSignature;
use crate::ast::typed::TypedDecl;
use crate::ast::typed::TypedExpression;
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum LinkError {
    DuplicateModule(String),
    ModuleNotFound(String),
    DuplicateSymbol(String),
    UnresolvedSymbols(String, Vec<String>),
    CircularDependency,
    EntryPointNotFound(String, String),
    InvalidReference(String),
}

impl WijError for LinkError {
    fn span(&self) -> Option<crate::Span> {
        None
    }

    fn reason(&self) -> String {
        match self {
            LinkError::DuplicateModule(name) => format!("Duplicate module: {}", name),
            LinkError::ModuleNotFound(name) => format!("Module not found: {}", name),
            LinkError::DuplicateSymbol(name) => format!("Duplicate symbol: {}", name),
            LinkError::UnresolvedSymbols(module, symbols) => {
                format!("Unresolved symbols in module {}: {:?}", module, symbols)
            }
            LinkError::CircularDependency => "Circular dependency detected".to_string(),
            LinkError::EntryPointNotFound(module, func) => {
                format!("Entry point {}::{} not found", module, func)
            }
            LinkError::InvalidReference(name) => format!("Invalid reference: {}", name),
        }
    }

    fn notes(&self) -> Vec<(String, crate::Span)> {
        vec![]
    }
}

#[derive(Debug, Default)]
struct Linker {
    modules: HashMap<String, Module>,
    // dependency graph for topological sorting
    dependencies: HashMap<String, HashSet<String>>,

    // global symbol table across all modules
    global_symbols: HashMap<String, (String, TypedDecl)>, // symbol_name -> (module_name, decl)

    // type definitions across modules
    global_types: HashMap<String, (String, Type)>, // type_name -> (module_name, type)

    // external dependencies that need to be resolved
    unresolved_externals: HashMap<String, Vec<(String, FunctionSignature)>>, // module -> externals

    // entry point specification
    entry_point: Option<(String, String)>,
}

impl Linker {
    pub fn set_entry_point(
        &mut self,
        module_name: String,
        function_name: String,
    ) -> Result<(), LinkError> {
        if !self.modules.contains_key(&module_name) {
            return Err(LinkError::ModuleNotFound(module_name));
        }

        let module = &self.modules[&module_name];
        let found = module.exports.iter().any(
            |decl| matches!(&decl.kind, DeclKind::Function { name, .. } if name == &function_name),
        );

        if !found {
            return Err(LinkError::EntryPointNotFound(module_name, function_name));
        }

        self.entry_point = Some((module_name, function_name));
        Ok(())
    }

    pub fn add_module(&mut self, module: Module) -> Result<(), LinkError> {
        let module_name = module.name.clone();
        if self.modules.contains_key(&module_name) {
            return Err(LinkError::DuplicateModule(module_name));
        }

        let mut deps = HashSet::new();
        self.extract_dependencies(&module, &mut deps)?;

        self.dependencies.insert(module_name.clone(), deps);
        self.modules.insert(module_name, module);

        Ok(())
    }

    fn extract_dependencies(
        &self,
        module: &Module,
        deps: &mut HashSet<String>,
    ) -> Result<(), LinkError> {
        todo!()
    }

    fn resolve_dependencies(&mut self) -> Result<(), LinkError> {
        todo!()
    }
}
