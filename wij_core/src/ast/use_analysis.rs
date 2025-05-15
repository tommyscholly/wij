use std::fs;
use std::io::Result;
use std::path::PathBuf;

use super::{Declaration, Path, Spanned};

pub fn resolve_file_from_path(file_path_base: &str, core_path: &str, mut path: Path) -> String {
    assert!(!path.is_empty());
    let mut base = String::from(file_path_base);
    if &path[0] == "core" {
        base = String::from(core_path);
        path.remove(0);
    }

    for segment in path {
        base.push('/');
        base.push_str(&segment);
    }

    base
}

pub fn files_in_module(module_path: &str) -> Result<Vec<PathBuf>> {
    let mut files = vec![];
    for file in fs::read_dir(module_path)? {
        let file = file?;
        let path = file.path();
        if path.is_file() {
            files.push(path);
        }
    }

    Ok(files)
}

pub fn extract_module_uses(decls: &Vec<Spanned<Declaration>>) -> Vec<Path> {
    let mut uses = vec![];
    for decl in decls {
        #[allow(clippy::single_match)]
        match &decl.0 {
            Declaration::Use(name) => {
                uses.push(name.clone());
            }
            _ => {}
        }
    }

    uses
}
