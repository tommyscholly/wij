use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::{
    Parser, Span, WijError,
    ast::{
        Declaration, Path as ASTPath,
        typed::{DeclKind as TypedDeclKind, Module, TypeChecker, TypedDecl},
        use_analysis::{extract_module_uses, files_in_module},
    },
    tokenize,
};

#[derive(Debug, Clone)]
pub enum CompilerErrorKind {
    ModuleNotFound(String),
    CircularDependency(Vec<String>),
    DuplicateModule(String),
    InvalidModulePath(String),
    ParseError(String),
    TypeError(String),
    IoError(String),
}

impl std::fmt::Display for CompilerErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilerErrorKind::ModuleNotFound(name) => write!(f, "Module not found: {}", name),
            CompilerErrorKind::CircularDependency(cycle) => {
                write!(f, "Circular dependency detected: {}", cycle.join(" -> "))
            }
            CompilerErrorKind::DuplicateModule(name) => write!(f, "Duplicate module: {}", name),
            CompilerErrorKind::InvalidModulePath(path) => {
                write!(f, "Invalid module path: {}", path)
            }
            CompilerErrorKind::ParseError(msg) => write!(f, "Parse error: {}", msg),
            CompilerErrorKind::TypeError(msg) => write!(f, "Type error: {}", msg),
            CompilerErrorKind::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

#[derive(Debug)]
pub struct CompilerError {
    kind: CompilerErrorKind,
    span: Option<Span>,
}

impl CompilerError {
    pub fn new(kind: CompilerErrorKind, span: Option<Span>) -> Self {
        Self { kind, span }
    }
}

impl WijError for CompilerError {
    fn span(&self) -> Option<Span> {
        self.span.clone()
    }

    fn reason(&self) -> String {
        self.kind.to_string()
    }

    fn notes(&self) -> Vec<(String, Span)> {
        Vec::new()
    }
}

pub type CompilerResult<T> = Result<T, CompilerError>;

/// Represents a discovered module with its metadata
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub name: String,
    pub path: PathBuf,
    pub files: Vec<PathBuf>,
    pub imports: Vec<Import>,
    pub exports: Vec<ExportedSymbol>,
    pub dependencies: HashSet<String>,
    pub parsed: bool,
    pub typed: bool,
}

/// Represents an import statement in a module
#[derive(Debug, Clone)]
pub enum Import {
    /// `use core:fmt;` - imports all public symbols
    Glob(String),
    /// `use core::fmt::println;` - imports specific symbol
    Specific(String, String),
    /// `use core::fmt::{println, print};` - imports multiple symbols
    Multiple(String, Vec<String>),
}

/// Represents an exported symbol from a module
#[derive(Debug, Clone)]
pub struct ExportedSymbol {
    pub name: String,
    pub kind: SymbolKind,
    pub qualified_name: String, // module::symbol
}

#[derive(Debug, Clone)]
pub enum SymbolKind {
    Function,
    Type,
    Record,
    Enum,
}

/// The main compiler with discovery phase
pub struct Compiler {
    /// Base directory for the project
    base_path: PathBuf,
    /// Core library path
    core_path: PathBuf,
    /// Discovered modules indexed by name
    modules: HashMap<String, ModuleInfo>,
    /// Dependency graph for build ordering
    dependency_graph: HashMap<String, HashSet<String>>,
    /// Global symbol table across all modules
    global_symbols: HashMap<String, ExportedSymbol>,
}

impl Compiler {
    pub fn new(base_path: PathBuf, core_path: PathBuf) -> Self {
        Self {
            base_path,
            core_path,
            modules: HashMap::new(),
            dependency_graph: HashMap::new(),
            global_symbols: HashMap::new(),
        }
    }

    /// Phase 1: Discover all modules in the project
    pub fn discover_modules(&mut self) -> CompilerResult<()> {
        // Start by discovering the main module and core modules
        self.discover_module_recursive("main", &self.base_path.clone())?;
        // self.discover_core_modules()?;

        Ok(())
    }

    /// Discover modules starting from a specific file
    pub fn discover_from_file(&mut self, file_path: &Path) -> CompilerResult<()> {
        // Discover the main file as "main" module
        self.discover_module_recursive("main", file_path)?;
        // self.discover_core_modules()?;

        Ok(())
    }

    /// Recursively discover modules starting from a root
    fn discover_module_recursive(
        &mut self,
        module_name: &str,
        module_path: &Path,
    ) -> CompilerResult<()> {
        if self.modules.contains_key(module_name) {
            return Ok(()); // Already discovered
        }

        if !module_path.exists() {
            return Err(CompilerError::new(
                CompilerErrorKind::ModuleNotFound(module_name.to_string()),
                None,
            ));
        }

        // Handle single file case (for main file compilation)
        if module_path.is_file() && module_path.extension().and_then(|s| s.to_str()) == Some("wij")
        {
            let mut module_info = ModuleInfo {
                name: module_name.to_string(),
                path: module_path.to_path_buf(),
                files: vec![module_path.to_path_buf()],
                imports: Vec::new(),
                exports: Vec::new(),
                dependencies: HashSet::new(),
                parsed: false,
                typed: false,
            };

            // Extract imports from the single file
            self.extract_imports(&mut module_info)?;
            self.modules.insert(module_name.to_string(), module_info);

            // Recursively discover imported modules
            let imports = self.modules[module_name].imports.clone();
            for import in imports {
                let dep_module_name = match &import {
                    Import::Glob(name) | Import::Specific(name, _) | Import::Multiple(name, _) => {
                        name
                    }
                };

                if !self.modules.contains_key(dep_module_name) {
                    let dep_path = self.resolve_module_path(dep_module_name)?;
                    self.discover_module_recursive(dep_module_name, &dep_path)?;
                }

                // Add dependency
                self.dependency_graph
                    .entry(module_name.to_string())
                    .or_insert_with(HashSet::new)
                    .insert(dep_module_name.clone());
            }

            return Ok(());
        }
        // Get all .wij files in the module directory
        let files = files_in_module(&module_path.to_string_lossy())
            .map_err(|e| CompilerError::new(CompilerErrorKind::IoError(e.to_string()), None))?;

        let wij_files: Vec<PathBuf> = files
            .into_iter()
            .filter(|f| f.extension().and_then(|s| s.to_str()) == Some("wij"))
            .collect();

        if wij_files.is_empty() {
            return Err(CompilerError::new(
                CompilerErrorKind::ModuleNotFound(format!(
                    "No .wij files found in {}",
                    module_name
                )),
                None,
            ));
        }

        let mut module_info = ModuleInfo {
            name: module_name.to_string(),
            path: module_path.to_path_buf(),
            files: wij_files,
            imports: Vec::new(),
            exports: Vec::new(),
            dependencies: HashSet::new(),
            parsed: false,
            typed: false,
        };

        // Parse files to extract imports without full parsing
        self.extract_imports(&mut module_info)?;

        // Add to modules map
        self.modules.insert(module_name.to_string(), module_info);

        // Recursively discover imported modules
        let imports = self.modules[module_name].imports.clone();
        for import in imports {
            let dep_module_name = match &import {
                Import::Glob(name) | Import::Specific(name, _) | Import::Multiple(name, _) => name,
            };

            if !self.modules.contains_key(dep_module_name) {
                let dep_path = self.resolve_module_path(dep_module_name)?;
                self.discover_module_recursive(dep_module_name, &dep_path)?;
            }

            // build our dependency graph to allow us to determine the order of compilation
            self.dependency_graph
                .entry(module_name.to_string())
                .or_insert_with(HashSet::new)
                .insert(dep_module_name.clone());
        }

        Ok(())
    }

    /// Unsure if we want to be compiling all of core on every single program
    /// Probably better to wait until we have a better understanding of the core modules
    fn discover_core_modules(&mut self) -> CompilerResult<()> {
        let core_dir = &self.core_path;
        if !core_dir.exists() {
            return Ok(()); // No core modules
        }

        // Read core directory
        let entries = fs::read_dir(core_dir)
            .map_err(|e| CompilerError::new(CompilerErrorKind::IoError(e.to_string()), None))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| CompilerError::new(CompilerErrorKind::IoError(e.to_string()), None))?;

            if entry
                .file_type()
                .map_err(|e| CompilerError::new(CompilerErrorKind::IoError(e.to_string()), None))?
                .is_dir()
            {
                let module_name = format!("core:{}", entry.file_name().to_string_lossy());
                if !self.modules.contains_key(&module_name) {
                    self.discover_module_recursive(&module_name, &entry.path())?;
                }
            }
        }

        Ok(())
    }

    /// Extract import statements from a module without full parsing
    fn extract_imports(&self, module_info: &mut ModuleInfo) -> CompilerResult<()> {
        for file_path in &module_info.files {
            let content = fs::read_to_string(file_path)
                .map_err(|e| CompilerError::new(CompilerErrorKind::IoError(e.to_string()), None))?;

            // Quick tokenization and parsing just for imports
            let lexer = tokenize(&content);
            let mut tokens = Vec::new();
            for token_result in lexer {
                match token_result {
                    Ok(token) => tokens.push(token),
                    Err(e) => {
                        // If we get an "Unexpected end of input" error, that's normal at end of file
                        if e.reason().contains("Unexpected end of input") {
                            break;
                        }
                        return Err(CompilerError::new(
                            CompilerErrorKind::ParseError(e.reason()),
                            e.span(),
                        ));
                    }
                }
            }

            let mut parser = Parser::new(tokens.into());

            // Parse declarations and extract uses
            let mut declarations = Vec::new();
            for decl_result in &mut parser {
                match decl_result {
                    Ok(decl) => {
                        declarations.push(decl);
                    }
                    Err(e) => {
                        // For discovery phase, we'll be more lenient with parse errors
                        // and continue with what we can parse
                        if e.reason().contains("EndOfInput")
                            || e.reason().contains("Unexpected end of input")
                        {
                            break;
                        }
                        // For other parse errors, we can optionally continue or fail
                        // For now, let's continue and see what imports we can extract
                        break;
                    }
                }
            }

            let uses = extract_module_uses(&declarations);
            for use_path in uses {
                let import = self.convert_ast_path_to_import(use_path);
                module_info.imports.push(import);
            }
        }

        Ok(())
    }

    /// Convert AST path to Import enum
    fn convert_ast_path_to_import(&self, path: ASTPath) -> Import {
        // For now, treat all imports as glob imports
        // This is where you'd extend to support qualified imports like `fmt:println`
        let module_name = path.join(":");
        println!("Importing module: {}", module_name);
        Import::Glob(module_name)
    }

    /// Resolve module name to filesystem path
    fn resolve_module_path(&self, module_name: &str) -> CompilerResult<PathBuf> {
        let path_segments: Vec<&str> = module_name.split(":").collect();

        let base_path = if path_segments[0] == "core" {
            &self.core_path
        } else {
            &self.base_path
        };

        let mut path = base_path.clone();
        for segment in &path_segments[1..] {
            // Skip first segment if it's "core"
            path.push(segment);
        }

        Ok(path)
    }

    /// Phase 2: Build dependency order using topological sort
    pub fn resolve_build_order(&self) -> CompilerResult<Vec<String>> {
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();
        let mut order = Vec::new();

        for module_name in self.modules.keys() {
            if !visited.contains(module_name) {
                self.topological_sort(module_name, &mut visited, &mut visiting, &mut order)?;
            }
        }

        Ok(order)
    }

    /// Topological sort helper for dependency resolution
    fn topological_sort(
        &self,
        module: &str,
        visited: &mut HashSet<String>,
        visiting: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) -> CompilerResult<()> {
        println!("Debug: topological_sort for module: {}", module);

        if let Some(deps) = self.dependency_graph.get(module) {
            println!("Debug: {} depends on: {:?}", module, deps);
        } else {
            println!("Debug: {} has no dependencies", module);
        }

        if visiting.contains(module) {
            // Circular dependency detected
            let mut cycle = visiting.iter().cloned().collect::<Vec<_>>();
            cycle.push(module.to_string());
            return Err(CompilerError::new(
                CompilerErrorKind::CircularDependency(cycle),
                None,
            ));
        }

        if visited.contains(module) {
            println!("Debug: {} already visited", module);
            return Ok(());
        }

        visiting.insert(module.to_string());

        if let Some(deps) = self.dependency_graph.get(module) {
            for dep in deps {
                println!("Debug: Processing dependency {} for {}", dep, module);
                self.topological_sort(dep, visited, visiting, order)?;
            }
        }

        visiting.remove(module);
        visited.insert(module.to_string());
        order.push(module.to_string());
        println!("Debug: Added {} to build order", module);

        Ok(())
    }

    /// Phase 3: Parse and type-check modules in dependency order
    pub fn compile_modules(&mut self) -> CompilerResult<Vec<Module>> {
        let build_order = self.resolve_build_order()?;
        let mut typed_modules = Vec::new();
        let mut compiled_modules: HashMap<String, Module> = HashMap::new();

        for module_name in build_order {
            println!("Debug: Compiling module: {}", module_name);
            let module_info = self.modules.get(&module_name).unwrap().clone();

            // Parse module
            println!("Debug: Parsing module: {}", module_name);
            let parsed_module = self.parse_module(&module_info)?;
            println!(
                "Debug: Parsed {} declarations for module: {}",
                parsed_module.len(),
                module_name
            );

            // Collect imports from already compiled modules
            println!("Debug: Collecting imports for module: {}", module_name);
            let imports = self.collect_imports(&module_info, &compiled_modules)?;
            println!(
                "Debug: Collected {} imports for module: {}",
                imports.len(),
                module_name
            );

            // Type check
            println!("Debug: Type checking module: {}", module_name);
            let typed_module = self.type_check_module(parsed_module, imports)?;
            println!("Debug: Successfully compiled module: {}", module_name);

            // Update global symbols
            self.update_global_symbols(&typed_module);

            compiled_modules.insert(module_name, typed_module.clone());
            typed_modules.insert(0, typed_module);
        }

        Ok(typed_modules)
    }

    /// Parse a single module
    fn parse_module(
        &self,
        module_info: &ModuleInfo,
    ) -> CompilerResult<Vec<crate::ast::Spanned<Declaration>>> {
        let mut all_decls = Vec::new();

        for file_path in &module_info.files {
            println!("Debug: Parsing file: {}", file_path.display());
            let content = fs::read_to_string(file_path)
                .map_err(|e| CompilerError::new(CompilerErrorKind::IoError(e.to_string()), None))?;

            let lexer = tokenize(&content);
            let mut tokens = Vec::new();
            for token_result in lexer {
                match token_result {
                    Ok(token) => tokens.push(token),
                    Err(e) => {
                        if e.reason().contains("Unexpected end of input") {
                            break;
                        }
                        println!(
                            "Debug: Tokenization error in {}: {}",
                            file_path.display(),
                            e.reason()
                        );
                        return Err(CompilerError::new(
                            CompilerErrorKind::ParseError(e.reason()),
                            e.span(),
                        ));
                    }
                }
            }

            println!(
                "Debug: Got {} tokens from {}",
                tokens.len(),
                file_path.display()
            );
            let mut parser = Parser::new(tokens.into());

            for decl_result in &mut parser {
                match decl_result {
                    Ok(decl) => {
                        println!("Debug: Parsed declaration: {:?}", decl.0.decl);
                        all_decls.push(decl);
                    }
                    Err(e) => {
                        println!(
                            "Debug: Parse error in {}: {} at {:?}",
                            file_path.display(),
                            e.reason(),
                            e.span()
                        );
                        if e.reason().contains("EndOfInput")
                            || e.reason().contains("Unexpected end of input")
                        {
                            break;
                        }
                        return Err(CompilerError::new(
                            CompilerErrorKind::ParseError(e.reason()),
                            e.span(),
                        ));
                    }
                }
            }
            println!(
                "Debug: Finished parsing {}, got {} declarations",
                file_path.display(),
                all_decls.len()
            );
        }

        Ok(all_decls)
    }

    /// Collect imports for a module
    fn collect_imports(
        &self,
        module_info: &ModuleInfo,
        compiled_modules: &HashMap<String, Module>,
    ) -> CompilerResult<Vec<TypedDecl>> {
        let mut imports = Vec::new();

        println!(
            "Debug: Module {} has {} import statements",
            module_info.name,
            module_info.imports.len()
        );
        for import in &module_info.imports {
            let import_module_name = match import {
                Import::Glob(name) | Import::Specific(name, _) | Import::Multiple(name, _) => name,
            };

            println!("Debug: Looking for imported module: {}", import_module_name);
            println!(
                "Debug: Available compiled modules: {:?}",
                compiled_modules.keys().collect::<Vec<_>>()
            );

            if let Some(imported_module) = compiled_modules.get(import_module_name) {
                let export_names = imported_module
                    .exports
                    .iter()
                    .map(|decl| decl.name())
                    .collect::<Vec<_>>();
                println!(
                    "Debug: Found imported module {}, it exports {:?}",
                    import_module_name, export_names
                );
                match import {
                    Import::Glob(_) => {
                        // Import all public symbols
                        imports.extend(imported_module.exports.clone());
                        println!(
                            "Debug: Imported {} symbols from {}",
                            imported_module.exports.len(),
                            import_module_name
                        );
                    }
                    Import::Specific(_, symbol) => {
                        // Import specific symbol
                        if let Some(exported_decl) = imported_module
                            .exports
                            .iter()
                            .find(|decl| decl.name() == Some(symbol))
                        {
                            imports.push(exported_decl.clone());
                        }
                    }
                    Import::Multiple(_, symbols) => {
                        // Import multiple specific symbols
                        for symbol in symbols {
                            if let Some(exported_decl) = imported_module
                                .exports
                                .iter()
                                .find(|decl| decl.name() == Some(symbol))
                            {
                                imports.push(exported_decl.clone());
                            }
                        }
                    }
                }
            } else {
                println!(
                    "Debug: Could not find imported module: {}",
                    import_module_name
                );
            }
        }

        Ok(imports)
    }

    /// Type check a parsed module
    fn type_check_module(
        &self,
        parsed_decls: Vec<crate::ast::Spanned<Declaration>>,
        imports: Vec<TypedDecl>,
    ) -> CompilerResult<Module> {
        println!(
            "Debug: Type checking module with {} declarations and {} imports",
            parsed_decls.len(),
            imports.len()
        );

        let type_checker = TypeChecker::new(
            parsed_decls,
            imports,
            HashMap::new(), // comptime_fns
            HashMap::new(), // monomorphic_fns
        )
        .map_err(|e| {
            println!("Debug: TypeChecker::new failed: {}", e.reason());
            CompilerError::new(CompilerErrorKind::TypeError(e.reason()), e.span())
        })?;

        println!("Debug: TypeChecker created successfully, producing module...");
        type_checker.produce_module().map_err(|e| {
            println!("Debug: produce_module failed: {}", e.reason());
            CompilerError::new(CompilerErrorKind::TypeError(e.reason()), e.span())
        })
    }

    /// Update global symbol table with exports from a module
    fn update_global_symbols(&mut self, module: &Module) {
        for export in &module.exports {
            if let Some(symbol_name) = export.name() {
                let kind = match &export.kind {
                    TypedDeclKind::Function { .. } => SymbolKind::Function,
                    TypedDeclKind::Record { .. } => SymbolKind::Record,
                    TypedDeclKind::Enum { .. } => SymbolKind::Enum,
                    _ => continue,
                };

                let exported_symbol = ExportedSymbol {
                    name: symbol_name.to_string(),
                    kind,
                    qualified_name: format!("{}::{}", module.name, symbol_name),
                };

                self.global_symbols
                    .insert(exported_symbol.qualified_name.clone(), exported_symbol);
            }
        }
    }

    /// Get all discovered modules
    pub fn modules(&self) -> &HashMap<String, ModuleInfo> {
        &self.modules
    }

    /// Get global symbol table
    pub fn global_symbols(&self) -> &HashMap<String, ExportedSymbol> {
        &self.global_symbols
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_compiler_creation() {
        let base_path = env::current_dir().unwrap();
        let core_path = base_path.join("core");
        let compiler = Compiler::new(base_path, core_path);

        assert!(compiler.modules.is_empty());
        assert!(compiler.global_symbols.is_empty());
    }
}
