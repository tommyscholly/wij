use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
    process::{Command, exit},
};

use ariadne::{ColorGenerator, Label, Report, ReportKind, Source};
use clap::Parser as Clap;
use rand::{Rng, rng};

use wij_codegen::{Backend, CodegenOptions, codegen};
use wij_core::{
    Compiler, DeclKind, Graphviz, Module, Parser, Program, TypeChecker, WijError, build_ssa,
    tokenize, use_analysis,
};

#[derive(Clap)]
struct Options {
    file: String,
    #[clap(short, long, required = true)]
    core_path: String,
    #[clap(short, long)]
    lex: bool,
    #[clap(short, long)]
    parse: bool,
    #[clap(short, long)]
    tychk: bool,
    #[clap(short, long)]
    debug: bool,
    #[clap(short, long)]
    graphviz: bool,
}

fn report_error(file: &str, contents: &str, top_level_msg: &str, e: impl WijError) {
    let mut rand_state = [0u16; 3];
    rng().fill(&mut rand_state);
    let mut colors = ColorGenerator::from_state(rand_state, 0.5);

    let span = match e.span() {
        Some(span) => span,
        None => 0..0,
    };
    let mut report =
        Report::build(ReportKind::Error, (file, span.clone())).with_message(top_level_msg);

    if e.notes().is_empty() {
        report = report.with_label(
            Label::new((file, span))
                .with_message(e.reason())
                .with_color(colors.next()),
        );
    } else {
        report = report.with_note(e.reason());
    }

    for (msg, span) in e.notes() {
        report = report.with_label(
            Label::new((file, span))
                .with_message(msg)
                .with_color(colors.next()),
        );
    }

    report
        .finish()
        .print((file, Source::from(contents)))
        .unwrap();
}

// this is a special type alias, where the directly compiled file is always first
// and any dependent modules are appended
pub type ResultingModules = Vec<Module>;

fn compile_with_discovery(options: &Options) -> Option<ResultingModules> {
    let file_path = PathBuf::from(&options.file);
    if !file_path.exists() {
        eprintln!("File not found: {}", file_path.display());
        return None;
    }

    let directory = file_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let core_path = PathBuf::from(&options.core_path);

    let mut compiler = Compiler::new();

    let root_module = match compiler.discover_from_directory(directory, &core_path) {
        Ok(module) => module,
        Err(e) => {
            eprintln!("Discovery error: {}", e.reason());
            return None;
        }
    };

    if options.debug {
        let all_modules = root_module.all_modules();
        println!("Discovered {} modules", all_modules.len());
        println!("Module tree:");
        println!("{}", compiler.get_module_summary(&root_module));
    }

    if compiler.has_circular_dependencies() {
        eprintln!("Circular dependencies detected!");
        return None;
    }

    let module_order = match compiler.get_compilation_order(&root_module) {
        Ok(order) => order,
        Err(e) => {
            eprintln!("Failed to determine compilation order: {}", e.reason());
            return None;
        }
    };

    if options.debug {
        println!(
            "Compilation order: {:?}",
            module_order.iter().map(|m| &m.name).collect::<Vec<_>>()
        );
    }

    // compile modules in dependency order
    let mut compiled_modules = Vec::new();
    let mut compiled_set = HashSet::new();
    let mut compiled_modules_map = HashMap::new();

    for discovered_module in module_order {
        if options.debug {
            println!("Compiling module: {}", discovered_module.name);
        }

        let mut combined_module = Module::new(discovered_module.name.clone());

        // compile all files in this module
        // todo: only compile files that are used within the module
        for file_path in &discovered_module.files {
            if let Some(file_modules) = compile_file(
                file_path.to_str().unwrap(),
                options,
                &mut compiled_set,
                &compiled_modules_map,
            ) {
                for file_module in file_modules {
                    combined_module.combine(file_module);
                }
            } else {
                eprintln!("Failed to compile file: {}", file_path.display());
                return None;
            }
        }

        // store compiled module for subsequent modules
        // check if this is a core module and adjust the key accordingly
        let storage_key = if discovered_module
            .files
            .iter()
            .any(|f| f.starts_with(&options.core_path))
        {
            format!("core:{}", discovered_module.name)
        } else {
            discovered_module.name.clone()
        };

        if options.debug {
            println!("Storing compiled module with key: {}", storage_key);
            println!("  Module has {} exports", combined_module.exports.len());
        }

        compiled_modules_map.insert(storage_key, combined_module.clone());
        compiled_modules.push(combined_module);
    }

    if options.debug {
        println!("Compiled {} modules successfully", compiled_modules.len());
    }

    Some(compiled_modules)
}

fn compile_file(
    file: &str,
    options: &Options,
    compiled: &mut HashSet<String>,
    compiled_modules: &HashMap<String, Module>,
) -> Option<Vec<Module>> {
    if options.debug {
        println!("Compiling file: {}", file);
    }
    if compiled.contains(file) {
        return None;
    }
    compiled.insert(file.to_string());

    let src = match std::fs::read_to_string(file) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Failed to read file {}: {}", file, e);
            return None;
        }
    };
    let src = src.trim();

    let tokens = tokenize(src)
        .map(|t| match t {
            Ok(t) => t,
            Err(e) => {
                report_error(file, src, "Lex Error", e);
                exit(1);
            }
        })
        .collect();

    if options.lex {
        println!("{:#?}", tokens);
        return None;
    }

    let parser = Parser::new(tokens);
    let mut prog = Vec::new();
    for decl in parser {
        match decl {
            Ok(decl) => prog.push(decl),
            Err(e) => {
                report_error(file, src, "Parse Error", e);
                return None;
            }
        }
    }

    if options.parse {
        println!("{:#?}", prog);
        return None;
    }

    let module_uses = use_analysis::extract_module_uses(&prog);
    let mut imports = Vec::new();

    if options.debug {
        println!(
            "Available compiled modules: {:?}",
            compiled_modules.keys().collect::<Vec<_>>()
        );
        println!("Required module imports: {:?}", module_uses);
    }

    for module_import in module_uses {
        let module_name = module_import.join(":");
        // use the last segment of the module path as the namespace prefix
        let namespace_prefix = module_import.last().unwrap_or(&module_name);

        if options.debug {
            println!("Looking for module: {}", module_name);
            println!("Using namespace prefix: {}", namespace_prefix);
        }

        if let Some(compiled_module) = compiled_modules.get(&module_name) {
            if options.debug {
                println!(
                    "Found module {} with {} exports",
                    module_name,
                    compiled_module.exports.len()
                );
                println!("  Exports:");
                for export in &compiled_module.exports {
                    println!("    - {:?}", export);
                }
            }

            // creating namespaced imports for qualified function calls
            for export in &compiled_module.exports {
                let mut namespaced_export = export.clone();
                if let DeclKind::Function { name, .. } = &mut namespaced_export.kind {
                    *name = format!("{}:{}", namespace_prefix, name);
                }
                imports.push(namespaced_export);
            }
        } else {
            eprintln!("Module {} not found in compiled modules", module_name);
            if options.debug {
                eprintln!(
                    "Available modules: {:?}",
                    compiled_modules.keys().collect::<Vec<_>>()
                );
            }
            return None;
        }
    }

    let type_checker = match TypeChecker::new(prog, imports) {
        Ok(type_checker) => type_checker,
        Err(e) => {
            report_error(file, src, "Type Error", e);
            return None;
        }
    };

    let module = match type_checker.produce_module() {
        Ok(module) => module,
        Err(e) => {
            println!("{e:?}");
            report_error(file, src, "Type Error", e);
            return None;
        }
    };

    if options.tychk {
        println!("{:#?}", module);
        return None;
    }

    Some(vec![module])
}

fn main() {
    let options = Options::parse();

    let modules = compile_with_discovery(&options);

    if let Some(mut modules) = modules {
        // println!("Modules: {:#?}", modules);
        modules.reverse();
        let ssa_mod: Program = build_ssa(modules);

        if options.graphviz {
            let dot_content = ssa_mod.dot();
            let output_path = &ssa_mod.name;
            fs::write(format!("{}.dot", output_path), dot_content)
                .expect("Failed to write DOT file");

            Command::new("dot")
                .args([
                    "-Tpng",
                    &format!("{}.dot", output_path),
                    "-o",
                    &format!("{}.png", output_path),
                ])
                .output()
                .expect("Failed to execute dot command");
        }

        let backend = Backend::Cranelift;
        let options = CodegenOptions::new(ssa_mod.name.clone(), backend);
        codegen(ssa_mod, options);
    }
}
