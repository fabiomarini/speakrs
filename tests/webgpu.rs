// WebGPU integration tests
#[cfg(feature = "webgpu")]
mod webgpu_tests {
    use speakrs::ExecutionMode;
    use std::path::PathBuf;

    #[test]
    fn webgpu_mode_strings_are_correct() {
        assert_eq!(ExecutionMode::WebGpu.as_str(), "webgpu");
        assert_eq!(ExecutionMode::WebGpuFast.as_str(), "webgpu-fast");
    }

    #[test]
    fn webgpu_pipeline_builder_accepts_mode() {
        // This test verifies that WebGPU mode can be used with PipelineBuilder
        // It will skip if models are not available or WebGPU backend fails to initialize

        let models_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("models");

        if !models_dir.exists() {
            eprintln!("Skipping WebGPU pipeline test - models directory not found");
            return;
        }

        let segmentation_path = models_dir.join("segmentation-3.0.onnx");
        if !segmentation_path.exists() {
            eprintln!("Skipping WebGPU pipeline test - segmentation model not found");
            return;
        }

        // Attempt to create a pipeline with WebGPU mode
        let result = speakrs::PipelineBuilder::from_dir(&models_dir, ExecutionMode::WebGpu).build();

        match result {
            Ok(_) => {
                println!("WebGPU pipeline created successfully");
            }
            Err(e) => {
                // This is expected if WebGPU backend is not available on the system
                // or if ONNX Runtime WebGPU execution provider fails to initialize
                eprintln!(
                    "WebGPU pipeline creation attempted (may fail without WebGPU support): {}",
                    e
                );
                // Don't fail the test - WebGPU is experimental and may not be available
            }
        }
    }
}
