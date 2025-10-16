// src/execution_provider.rs
// Defines execution providers and GPU capabilities

use crate::error::InferaError;
use std::fmt;

/// Represents different execution providers for ONNX models
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// CPU execution provider (always available)
    CPU,
    /// NVIDIA CUDA execution provider
    CUDA,
    /// AMD ROCm execution provider  
    ROCm,
    /// DirectML execution provider (Windows)
    DirectML,
    /// Apple CoreML execution provider (macOS)
    CoreML,
}

impl fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutionProvider::CPU => write!(f, "CPU"),
            ExecutionProvider::CUDA => write!(f, "CUDA"),
            ExecutionProvider::ROCm => write!(f, "ROCm"),
            ExecutionProvider::DirectML => write!(f, "DirectML"),
            ExecutionProvider::CoreML => write!(f, "CoreML"),
        }
    }
}

impl ExecutionProvider {
    /// Parse execution provider from string
    pub fn from_str(s: &str) -> Result<Self, InferaError> {
        match s.to_uppercase().as_str() {
            "CPU" => Ok(ExecutionProvider::CPU),
            "CUDA" => Ok(ExecutionProvider::CUDA),
            "ROCM" => Ok(ExecutionProvider::ROCm),
            "DIRECTML" => Ok(ExecutionProvider::DirectML),
            "COREML" => Ok(ExecutionProvider::CoreML),
            _ => Err(InferaError::InvalidExecutionProvider(s.to_string())),
        }
    }

    /// Check if this execution provider is available on the current system
    #[cfg(feature = "onnxruntime")]
    pub fn is_available(&self) -> bool {
        match self {
            ExecutionProvider::CPU => true, // CPU is always available
            ExecutionProvider::CUDA => Self::check_cuda_available(),
            ExecutionProvider::ROCm => Self::check_rocm_available(),
            ExecutionProvider::DirectML => Self::check_directml_available(),
            ExecutionProvider::CoreML => Self::check_coreml_available(),
        }
    }

    /// Stub when onnxruntime feature is disabled
    #[cfg(not(feature = "onnxruntime"))]
    pub fn is_available(&self) -> bool {
        matches!(self, ExecutionProvider::CPU)
    }

    /// Get ONNX Runtime provider name
    #[cfg(feature = "onnxruntime")]
    pub fn to_onnxruntime_name(&self) -> &'static str {
        match self {
            ExecutionProvider::CPU => "CPUExecutionProvider",
            ExecutionProvider::CUDA => "CUDAExecutionProvider",
            ExecutionProvider::ROCm => "ROCmExecutionProvider",
            ExecutionProvider::DirectML => "DmlExecutionProvider",
            ExecutionProvider::CoreML => "CoreMLExecutionProvider",
        }
    }

    /// Get all available execution providers on this system
    pub fn get_available_providers() -> Vec<ExecutionProvider> {
        let all_providers = vec![
            ExecutionProvider::CPU,
            ExecutionProvider::CUDA,
            ExecutionProvider::ROCm,
            ExecutionProvider::DirectML,
            ExecutionProvider::CoreML,
        ];
        
        all_providers
            .into_iter()
            .filter(|p| p.is_available())
            .collect()
    }

    #[cfg(feature = "onnxruntime")]
    fn check_cuda_available() -> bool {
        // Try to detect CUDA runtime
        std::process::Command::new("nvidia-smi")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    #[cfg(feature = "onnxruntime")]
    fn check_rocm_available() -> bool {
        // Try to detect ROCm runtime
        std::process::Command::new("rocm-smi")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    #[cfg(feature = "onnxruntime")]
    fn check_directml_available() -> bool {
        // DirectML is Windows-specific
        cfg!(target_os = "windows")
    }

    #[cfg(feature = "onnxruntime")]
    fn check_coreml_available() -> bool {
        // CoreML is macOS-specific
        cfg!(target_os = "macos")
    }
}

/// Represents performance metrics for an execution provider
#[derive(Debug, Clone)]
pub struct ProviderBenchmark {
    pub provider: ExecutionProvider,
    pub inference_time_ms: f64,
    pub memory_usage_mb: f64,
    pub successful: bool,
}

impl ProviderBenchmark {
    pub fn new(provider: ExecutionProvider) -> Self {
        Self {
            provider,
            inference_time_ms: 0.0,
            memory_usage_mb: 0.0,
            successful: false,
        }
    }
}