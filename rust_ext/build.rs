//! Build script for compiling Metal shaders.
//!
//! This script compiles .metal files into a .metallib at build time.
//! Metal toolchain is REQUIRED on macOS - build will fail if not available.

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Only compile on macOS
    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "macos" {
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").unwrap();
    let shader_dir = Path::new("shaders");

    // Check if Metal toolchain is available - REQUIRED on macOS
    let toolchain_check = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output();

    let toolchain_available = toolchain_check
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !toolchain_available {
        panic!(
            "Metal toolchain is REQUIRED but not available.\n\
             Please install the Metal toolchain:\n\
             \n\
             Option 1: Install Xcode command line tools\n\
               xcode-select --install\n\
             \n\
             Option 2: Download Metal toolchain directly\n\
               xcodebuild -downloadComponent MetalToolchain\n\
             \n\
             Option 3: Install full Xcode from the App Store\n"
        );
    }

    // Collect all .metal files
    let metal_files: Vec<_> = std::fs::read_dir(shader_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "metal"))
        .collect();

    if metal_files.is_empty() {
        panic!(
            "No .metal shader files found in shaders/ directory.\n\
             Metal shaders are required for this build."
        );
    }

    // Compile each .metal file to .air
    let mut air_files = Vec::new();

    for entry in &metal_files {
        let metal_path = entry.path();
        let stem = metal_path.file_stem().unwrap().to_str().unwrap();
        let air_path = Path::new(&out_dir).join(format!("{}.air", stem));

        println!("cargo:rerun-if-changed={}", metal_path.display());

        let result = Command::new("xcrun")
            .args([
                "-sdk", "macosx",
                "metal",
                "-c",
                "-target", "air64-apple-macos14.0",
                "-ffast-math",
                "-o", air_path.to_str().unwrap(),
                metal_path.to_str().unwrap(),
            ])
            .output();

        match result {
            Ok(output) if output.status.success() => {
                air_files.push(air_path);
            }
            Ok(output) => {
                panic!(
                    "Failed to compile shader {}:\n{}",
                    metal_path.display(),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Err(e) => {
                panic!("Error running metal compiler: {}", e);
            }
        }
    }

    if air_files.is_empty() {
        panic!("No shader files were compiled successfully.");
    }

    // Link all .air files into a single .metallib
    let metallib_path = Path::new(&out_dir).join("vllm_kernels.metallib");

    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib", "-o", metallib_path.to_str().unwrap()]);
    for air_file in &air_files {
        cmd.arg(air_file.to_str().unwrap());
    }

    match cmd.output() {
        Ok(output) if output.status.success() => {
            println!("cargo:rustc-env=METALLIB_PATH={}", metallib_path.display());
            println!("cargo:warning=Compiled {} shaders into metallib", air_files.len());
        }
        Ok(output) => {
            panic!(
                "Failed to create metallib:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Err(e) => {
            panic!("Error running metallib linker: {}", e);
        }
    }
}
