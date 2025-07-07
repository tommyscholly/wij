#![deny(unused, dead_code)]

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs,
    path::{Path, PathBuf},
};

use crate::{
    Module, ParseError, Parser, Span, WijError,
    ast::{ParseErrorKind, Path as ASTPath, typed::TypedDecl, use_analysis::extract_module_uses},
    tokenize,
};

pub struct CompilerError {
    inner_err: Box<dyn WijError>,
}

impl CompilerError {
    pub fn new(inner_err: impl WijError + 'static) -> Self {
        Self {
            inner_err: Box::new(inner_err),
        }
    }

    pub fn from_message(message: String) -> Self {
        Self {
            inner_err: Box::new(SimpleError::new(message)),
        }
    }
}

// Simple error type for basic error messages
// Todo: remove this
#[derive(Debug)]
struct SimpleError {
    message: String,
}

impl SimpleError {
    fn new(message: String) -> Self {
        Self { message }
    }
}

impl WijError for SimpleError {
    fn span(&self) -> Option<Span> {
        None
    }

    fn notes(&self) -> Vec<(String, Span)> {
        Vec::new()
    }

    fn reason(&self) -> String {
        self.message.clone()
    }
}

impl WijError for CompilerError {
    fn span(&self) -> Option<Span> {
        self.inner_err.span()
    }

    fn notes(&self) -> Vec<(String, Span)> {
        self.inner_err.notes()
    }

    fn reason(&self) -> String {
        self.inner_err.reason()
    }
}

#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub name: String,
    pub files: Vec<PathBuf>,
    pub dependencies: Vec<String>,
    pub imports: Vec<TypedDecl>,
    pub exports: Vec<TypedDecl>,
    pub children: HashMap<String, ModuleInfo>,
}

impl ModuleInfo {
    pub fn new(name: String) -> Self {
        Self {
            name,
            files: Vec::new(),
            dependencies: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            children: HashMap::new(),
        }
    }

    pub fn add_file(&mut self, file_path: PathBuf) {
        self.files.push(file_path);
    }

    pub fn add_dependency(&mut self, dep: String) {
        if !self.dependencies.contains(&dep) {
            self.dependencies.push(dep);
        }
    }

    pub fn add_child(&mut self, name: String, module: ModuleInfo) {
        self.children.insert(name, module);
    }

    pub fn get_child(&self, name: &str) -> Option<&ModuleInfo> {
        self.children.get(name)
    }

    pub fn all_modules(&self) -> Vec<&ModuleInfo> {
        let mut modules = vec![self];
        for child in self.children.values() {
            modules.extend(child.all_modules());
        }
        modules
    }
}

pub struct Compiler {
    modules: HashMap<String, ModuleInfo>,
    dependency_graph: HashMap<String, Vec<String>>,
    visited_modules: HashSet<String>,
    core_path: PathBuf,
}

pub type CompilerResult<T> = Result<T, CompilerError>;

impl Compiler {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            dependency_graph: HashMap::new(),
            visited_modules: HashSet::new(),
            core_path: PathBuf::new(),
        }
    }

    pub fn discover_from_directory(
        &mut self,
        directory: &Path,
        core_path: &Path,
    ) -> CompilerResult<ModuleInfo> {
        self.core_path = core_path.to_path_buf();

        // First pass: scan all .wij files and group by module declaration
        // Second pass: extract dependencies for each module
        // Third pass: build module tree based on dependencies

        self.scan_and_group_files(directory)?;
        self.extract_module_dependencies()?;
        self.build_module_tree()
    }

    pub fn discover_from_file(
        &mut self,
        file_path: &str,
        core_path: &str,
    ) -> CompilerResult<ModuleInfo> {
        let file_path = Path::new(file_path);
        let directory = file_path.parent().unwrap_or_else(|| Path::new("."));
        self.discover_from_directory(directory, Path::new(core_path))
    }

    fn scan_and_group_files(&mut self, directory: &Path) -> CompilerResult<()> {
        // Scan directory recursively and group files by their module declarations
        self.scan_directory_recursive(directory)?;

        // Also scan core directory
        let core_path = self.core_path.clone();
        if core_path.exists() {
            self.scan_directory_recursive(&core_path)?;
        }

        Ok(())
    }

    fn scan_directory_recursive(&mut self, directory: &Path) -> CompilerResult<()> {
        if !directory.exists() {
            return Ok(());
        }

        let entries = fs::read_dir(directory).map_err(|e| {
            CompilerError::new(ParseError {
                kind: ParseErrorKind::EndOfInput,
                span: Some(0..0),
                reason: Some(format!(
                    "Failed to read directory {}: {}",
                    directory.display(),
                    e
                )),
            })
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                CompilerError::new(ParseError {
                    kind: ParseErrorKind::EndOfInput,
                    span: Some(0..0),
                    reason: Some(format!("Failed to read directory entry: {}", e)),
                })
            })?;

            let path = entry.path();

            if path.is_dir() {
                self.scan_directory_recursive(&path)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("wij") {
                // Parse module declaration from .wij file
                let module_name = self.extract_module_declaration(&path)?;

                self.modules
                    .entry(module_name.clone())
                    .or_insert_with(|| ModuleInfo::new(module_name))
                    .add_file(path);
            }
        }

        Ok(())
    }

    /// Extract module declaration from a single file
    fn extract_module_declaration(&self, file_path: &Path) -> CompilerResult<String> {
        let content = fs::read_to_string(file_path).map_err(|e| {
            CompilerError::new(ParseError {
                kind: ParseErrorKind::EndOfInput,
                span: Some(0..0),
                reason: Some(format!(
                    "Failed to read file {}: {}",
                    file_path.display(),
                    e
                )),
            })
        })?;

        let token_stream = tokenize(content.trim());
        let mut tokens = VecDeque::new();

        for token in token_stream {
            let token = token.map_err(CompilerError::new)?;
            tokens.push_back(token);
        }

        let parser = Parser::new(tokens);

        // Look for the first declaration which should be a module declaration
        for decl in parser {
            let decl = decl.map_err(CompilerError::new)?;

            match &decl.0.decl {
                crate::ast::DeclKind::Module(module_name) => {
                    return Ok(module_name.clone());
                }
                _ => {
                    // If we hit a non-module declaration, this file doesn't have a module declaration
                    // and that is an error
                    break;
                }
            }
        }

        return Err(CompilerError::from_message(format!(
            "Expected module declaration in {}",
            file_path.display()
        )));
    }

    fn extract_module_dependencies(&mut self) -> CompilerResult<()> {
        let module_names: Vec<String> = self.modules.keys().cloned().collect();

        for module_name in module_names {
            let dependencies = self.extract_dependencies_for_module(&module_name)?;

            if let Some(module) = self.modules.get_mut(&module_name) {
                for dep in &dependencies {
                    module.add_dependency(dep.clone());
                }
            }

            self.dependency_graph.insert(module_name, dependencies);
        }

        Ok(())
    }

    fn extract_dependencies_for_module(&self, module_name: &str) -> CompilerResult<Vec<String>> {
        let module = self.modules.get(module_name).ok_or_else(|| {
            CompilerError::new(ParseError {
                kind: ParseErrorKind::MalformedExpression,
                span: Some(0..0),
                reason: Some(format!("Module not found: {}", module_name)),
            })
        })?;

        let mut dependencies = HashSet::new();

        for file_path in &module.files {
            let file_dependencies = self.extract_dependencies_from_file(file_path)?;
            dependencies.extend(file_dependencies);
        }

        Ok(dependencies.into_iter().collect())
    }

    fn extract_dependencies_from_file(&self, file_path: &Path) -> CompilerResult<Vec<String>> {
        let content = fs::read_to_string(file_path).map_err(|e| {
            CompilerError::new(ParseError {
                kind: ParseErrorKind::EndOfInput,
                span: Some(0..0),
                reason: Some(format!(
                    "Failed to read file {}: {}",
                    file_path.display(),
                    e
                )),
            })
        })?;

        let token_stream = tokenize(content.trim());
        let mut tokens = VecDeque::new();

        for token in token_stream {
            let token = token.map_err(CompilerError::new)?;
            tokens.push_back(token);
        }

        let parser = Parser::new(tokens);
        let mut declarations = Vec::new();

        for decl in parser {
            let decl = decl.map_err(CompilerError::new)?;
            declarations.push(decl);
        }

        let uses = extract_module_uses(&declarations);
        let dependencies: Vec<String> = uses
            .into_iter()
            .map(|path| self.resolve_module_from_path(&path))
            .collect();

        Ok(dependencies)
    }

    fn resolve_module_from_path(&self, path: &ASTPath) -> String {
        if path.is_empty() {
            return "unknown".to_string();
        }

        // For core modules, use just the last component (e.g., "fmt" not "core::fmt")
        if path[0] == "core" {
            return path.last().unwrap_or(&"unknown".to_string()).clone();
        }

        // For local modules, use just the last component
        path.last().expect("Path should not be empty").clone()
    }

    /// Build hierarchical module tree based on dependencies
    fn build_module_tree(&mut self) -> CompilerResult<ModuleInfo> {
        if !self.modules.contains_key("main") {
            return Err(CompilerError::new(ParseError {
                kind: ParseErrorKind::EndOfInput,
                span: Some(0..0),
                reason: Some("No main module found".to_string()),
            }));
        }

        self.build_tree_recursive("main")
    }

    fn build_tree_recursive(&mut self, module_name: &str) -> CompilerResult<ModuleInfo> {
        if self.visited_modules.contains(module_name) {
            // Return empty module to avoid infinite recursion
            // todo: find a better way of doing this, maybe an Result<Option<ModuleInfo>>
            return Ok(ModuleInfo::new(module_name.to_string()));
        }

        self.visited_modules.insert(module_name.to_string());

        let mut module = self.modules.remove(module_name).ok_or_else(|| {
            CompilerError::new(ParseError {
                kind: ParseErrorKind::MalformedExpression,
                span: Some(0..0),
                reason: Some(format!("Module not found: {}", module_name)),
            })
        })?;

        let dependencies = module.dependencies.clone();

        for dep_name in dependencies {
            if !self.visited_modules.contains(&dep_name) && self.modules.contains_key(&dep_name) {
                let child_module = self.build_tree_recursive(&dep_name)?;
                module.add_child(dep_name.clone(), child_module);
            }
        }

        Ok(module)
    }

    pub fn has_circular_dependencies(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for module in self.dependency_graph.keys() {
            if !visited.contains(module) {
                if self.has_cycle_util(module, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        false
    }

    fn has_cycle_util(
        &self,
        module: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(module.to_string());
        rec_stack.insert(module.to_string());

        if let Some(deps) = self.dependency_graph.get(module) {
            for dep in deps {
                if !visited.contains(dep) {
                    if self.has_cycle_util(dep, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(dep) {
                    return true;
                }
            }
        }

        rec_stack.remove(module);
        false
    }

    pub fn get_module_summary(&self, root_module: &ModuleInfo) -> String {
        let mut summary = String::new();
        self.build_module_summary(root_module, 0, &mut summary);
        summary
    }

    fn build_module_summary(&self, module: &ModuleInfo, depth: usize, summary: &mut String) {
        let indent = "  ".repeat(depth);
        summary.push_str(&format!(
            "{}module: {} ({} files)\n",
            indent,
            module.name,
            module.files.len()
        ));

        for file in &module.files {
            summary.push_str(&format!("{}  file: {}\n", indent, file.display()));
        }

        if !module.dependencies.is_empty() {
            summary.push_str(&format!(
                "{}  depends on: {}\n",
                indent,
                module.dependencies.join(", ")
            ));
        }

        for child_module in module.children.values() {
            self.build_module_summary(child_module, depth + 1, summary);
        }
    }

    /// topologically sort modules in compilation order
    pub fn get_compilation_order(
        &self,
        root_module: &ModuleInfo,
    ) -> CompilerResult<Vec<ModuleInfo>> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();

        self.topological_sort_recursive(root_module, &mut result, &mut visited, &mut visiting)?;
        Ok(result)
    }

    fn topological_sort_recursive(
        &self,
        module: &ModuleInfo,
        result: &mut Vec<ModuleInfo>,
        visited: &mut HashSet<String>,
        visiting: &mut HashSet<String>,
    ) -> CompilerResult<()> {
        if visiting.contains(&module.name) {
            return Err(CompilerError::from_message(format!(
                "Circular dependency detected involving module: {}",
                module.name
            )));
        }

        if visited.contains(&module.name) {
            return Ok(());
        }

        visiting.insert(module.name.clone());

        // first visit all dependencies
        for dep_name in &module.dependencies {
            if let Some(dep_module) = self.find_module_by_name(dep_name) {
                self.topological_sort_recursive(dep_module, result, visited, visiting)?;
            }
        }

        // then visit child modules
        for child in module.children.values() {
            self.topological_sort_recursive(child, result, visited, visiting)?;
        }

        visiting.remove(&module.name);
        visited.insert(module.name.clone());
        result.push(module.clone());

        Ok(())
    }

    fn find_module_by_name(&self, name: &str) -> Option<&ModuleInfo> {
        self.modules.get(name)
    }

    pub fn modules_to_vec(&self, root_module: &ModuleInfo) -> Vec<crate::Module> {
        let mut result = Vec::new();
        self.collect_modules_recursive(root_module, &mut result);
        result
    }

    fn collect_modules_recursive(&self, module: &ModuleInfo, result: &mut Vec<crate::Module>) {
        // convert module info to existing module
        let mut existing_module = Module::new(module.name.clone());

        for export in &module.exports {
            existing_module.exports.push(export.clone());
        }

        result.push(existing_module);

        for child in module.children.values() {
            self.collect_modules_recursive(child, result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_creation() {
        let mut module = ModuleInfo::new("test".to_string());
        module.add_file(PathBuf::from("test.wij"));
        module.add_dependency("core".to_string());

        assert_eq!(module.name, "test");
        assert_eq!(module.files.len(), 1);
        assert_eq!(module.dependencies.len(), 1);
    }
}
