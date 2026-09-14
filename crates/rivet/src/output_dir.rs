//! An output directory made for one run, and taken away again if the run
//! ends without putting anything in it.
//!
//! The CLI and the HTTP API make a job's output directory before the job
//! runs, so an unusable path fails before any work is done. A job refused
//! after that — by the encode pool's preflight, or a policy that leaves
//! nothing to encode on — used to leave the empty directory behind. A
//! [`CreatedDir`] records exactly the directories it made and, when dropped
//! before [`CreatedDir::keep`], removes them again, deepest first: each only
//! while it is empty, and never one that already existed.

use std::io;
use std::path::{Path, PathBuf};

/// The directories one [`CreatedDir::create`] call made. Dropping it removes
/// them again while they are empty; [`CreatedDir::keep`] keeps them.
#[derive(Debug)]
#[must_use = "dropping a CreatedDir removes the directories it made while they are empty"]
pub struct CreatedDir {
    /// Outermost first.
    created: Vec<PathBuf>,
}

impl CreatedDir {
    /// Make `dir` and any missing ancestors, as `std::fs::create_dir_all`
    /// does, recording which directories this call made. A directory that
    /// already exists records nothing, so it is never removed.
    pub fn create(dir: &Path) -> io::Result<Self> {
        let mut missing = Vec::new();
        let mut cur = Some(dir);
        while let Some(p) = cur {
            if p.as_os_str().is_empty() || p.exists() {
                break;
            }
            missing.push(p.to_path_buf());
            cur = p.parent();
        }
        let mut made = Self { created: Vec::new() };
        for p in missing.into_iter().rev() {
            match std::fs::create_dir(&p) {
                Ok(()) => made.created.push(p),
                // Made by someone else in the meantime: theirs, not ours.
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists && p.is_dir() => {}
                // `made` drops here and takes back what it made.
                Err(e) => return Err(e),
            }
        }
        // A no-op when every level now exists; the error it has always given
        // when `dir` is a file.
        std::fs::create_dir_all(dir)?;
        Ok(made)
    }

    /// The directories this call made, outermost first.
    pub fn created(&self) -> &[PathBuf] {
        &self.created
    }

    /// Keep every directory this call made, empty or not.
    pub fn keep(mut self) {
        self.created.clear();
    }
}

impl Drop for CreatedDir {
    fn drop(&mut self) {
        // Deepest first: a parent can only be empty once its child is gone.
        while let Some(p) = self.created.pop() {
            // `remove_dir` refuses a directory with anything in it. Every
            // directory above that one is then not empty either.
            if std::fs::remove_dir(&p).is_err() {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run that ends with nothing written takes back every directory it
    /// made, the whole missing chain, and nothing above it.
    #[test]
    fn a_run_that_writes_nothing_leaves_no_directory_it_made() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("a").join("b").join("c");
        let made = CreatedDir::create(&dir).unwrap();
        assert!(dir.is_dir());
        assert_eq!(
            made.created(),
            &[root.path().join("a"), root.path().join("a").join("b"), dir.clone()]
        );
        drop(made);
        assert!(!root.path().join("a").exists(), "the chain this run made is gone");
        assert!(root.path().is_dir(), "the directory that existed stays");
    }

    /// A directory that existed before the run is never removed, empty or not.
    #[test]
    fn a_directory_that_existed_is_never_removed() {
        let root = tempfile::tempdir().unwrap();
        let empty = root.path().join("empty");
        let full = root.path().join("full");
        std::fs::create_dir(&empty).unwrap();
        std::fs::create_dir(&full).unwrap();
        std::fs::write(full.join("keep.txt"), b"x").unwrap();
        for dir in [&empty, &full] {
            let made = CreatedDir::create(dir).unwrap();
            assert!(made.created().is_empty());
            drop(made);
            assert!(dir.is_dir(), "{}", dir.display());
        }
        assert_eq!(std::fs::read(full.join("keep.txt")).unwrap(), b"x");
    }

    /// Only the missing part of a chain is taken back.
    #[test]
    fn a_partly_existing_chain_loses_only_the_part_this_run_made() {
        let root = tempfile::tempdir().unwrap();
        let a = root.path().join("a");
        std::fs::create_dir(&a).unwrap();
        let made = CreatedDir::create(&a.join("b").join("c")).unwrap();
        assert_eq!(made.created(), &[a.join("b"), a.join("b").join("c")]);
        drop(made);
        assert!(a.is_dir());
        assert!(!a.join("b").exists());
    }

    /// Output in the directory keeps it, and every directory above it.
    #[test]
    fn a_directory_with_output_in_it_stays() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("a").join("b");
        let made = CreatedDir::create(&dir).unwrap();
        std::fs::write(dir.join("master.m3u8"), b"#EXTM3U\n").unwrap();
        drop(made);
        assert_eq!(std::fs::read(dir.join("master.m3u8")).unwrap(), b"#EXTM3U\n");
    }

    /// `keep` keeps even an empty directory.
    #[test]
    fn keep_keeps_an_empty_directory() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("a");
        CreatedDir::create(&dir).unwrap().keep();
        assert!(dir.is_dir());
    }

    /// A file where the directory should be is the error it always was, and
    /// the file is left alone.
    #[test]
    fn a_file_in_the_way_is_an_error_and_is_left_alone() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("f");
        std::fs::write(&file, b"x").unwrap();
        assert!(CreatedDir::create(&file).is_err());
        assert!(CreatedDir::create(&file.join("sub")).is_err());
        assert_eq!(std::fs::read(&file).unwrap(), b"x");
    }
}
