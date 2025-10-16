// src/model.rs
// Defines the internal representation of a model and the global model store.

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use crate::text::TextModelConfig;
use crate::execution_provider::ExecutionProvider;

#[cfg(feature = "tract")]
use tract_onnx::prelude::*;

#[cfg(feature = "onnxruntime")]
use onnxruntime::{Session, SessionBuilder, GraphOptimizationLevel};

/// Type alias for a Tract `SimplePlan`.
/// This represents a runnable, optimized ONNX model execution plan.
#[cfg(feature = "tract")]
pub(crate) type OnnxModelPlan =
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Enum to define the type of model and its specific configuration
#[derive(Debug, Clone)]
pub(crate) enum ModelType {
    /// Standard numerical model that expects float arrays
    Numerical,
    /// Text model that requires tokenization before inference
    Text(TextModelConfig),
}

/// Represents the execution engine for a model
pub(crate) enum ExecutionEngine {
    #[cfg(feature = "tract")]
    Tract(OnnxModelPlan),
    #[cfg(feature = "onnxruntime")]
    OnnxRuntime(Session),
}

/// Represents a loaded ONNX model, holding its execution plan and metadata.
pub(crate) struct OnnxModel {
    /// The execution engine (Tract or ONNX Runtime)
    pub engine: ExecutionEngine,
    /// The current execution provider
    pub execution_provider: ExecutionProvider,
    /// The model file path (for reloading with different providers)
    pub model_path: String,
    /// The shape of the model's input tensor. Dynamic dimensions are represented by -1.
    pub input_shape: Vec<i64>,
    /// The shape of the model's output tensor. Dynamic dimensions are represented by -1.
    pub output_shape: Vec<i64>,
    /// The user-defined name for the model.
    pub name: String,
    /// The type of model and its configuration
    pub model_type: ModelType,
}

impl OnnxModel {
    /// Run inference using the current execution engine
    #[cfg(feature = "tract")]
    pub fn run_inference(&self, inputs: Vec<tract_onnx::prelude::Tensor>) -> Result<tract_onnx::prelude::TVec<tract_onnx::prelude::TValue>, crate::error::InferaError> {
        match &self.engine {
            ExecutionEngine::Tract(model) => {
                let tvec_inputs: tract_onnx::prelude::TVec<tract_onnx::prelude::TValue> = inputs.into_iter().map(|t| t.into()).collect();
                model.run(tvec_inputs)
                    .map_err(|e| crate::error::InferaError::OnnxError(e.to_string()))
            }
            #[cfg(feature = "onnxruntime")]
            ExecutionEngine::OnnxRuntime(_session) => {
                // TODO: Convert tract tensors to onnxruntime format
                Err(crate::error::InferaError::FeatureNotEnabled("onnxruntime tensor conversion not implemented".to_string()))
            }
        }
    }

    /// Switch the execution provider for this model
    pub fn switch_provider(&mut self, new_provider: ExecutionProvider) -> Result<(), crate::error::InferaError> {
        if self.execution_provider == new_provider {
            return Ok(()); // Already using this provider
        }

        // Check if the new provider is available
        if !new_provider.is_available() {
            return Err(crate::error::InferaError::ExecutionProviderNotAvailable(new_provider.to_string()));
        }

        // Reload the model with the new provider
        match new_provider {
            ExecutionProvider::CPU => {
                #[cfg(feature = "tract")]
                {
                    self.engine = ExecutionEngine::Tract(Self::load_with_tract(&self.model_path)?);
                    self.execution_provider = new_provider;
                    Ok(())
                }
                #[cfg(not(feature = "tract"))]
                Err(crate::error::InferaError::FeatureNotEnabled("tract".to_string()))
            }
            _ => {
                #[cfg(feature = "onnxruntime")]
                {
                    self.engine = ExecutionEngine::OnnxRuntime(Self::load_with_onnxruntime(&self.model_path, &new_provider)?);
                    self.execution_provider = new_provider;
                    Ok(())
                }
                #[cfg(not(feature = "onnxruntime"))]
                Err(crate::error::InferaError::FeatureNotEnabled("onnxruntime".to_string()))
            }
        }
    }

    #[cfg(feature = "tract")]
    fn load_with_tract(model_path: &str) -> Result<OnnxModelPlan, crate::error::InferaError> {
        tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|e| crate::error::InferaError::OnnxError(e.to_string()))?
            .into_optimized()
            .map_err(|e| crate::error::InferaError::OnnxError(e.to_string()))?
            .into_runnable()
            .map_err(|e| crate::error::InferaError::OnnxError(e.to_string()))
    }

    #[cfg(feature = "onnxruntime")]
    fn load_with_onnxruntime(model_path: &str, provider: &ExecutionProvider) -> Result<Session, crate::error::InferaError> {
        let mut builder = SessionBuilder::new()
            .map_err(|e| crate::error::InferaError::OnnxRuntimeError(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| crate::error::InferaError::OnnxRuntimeError(e.to_string()))?;

        // Add the specific execution provider
        builder = match provider {
            ExecutionProvider::CPU => builder,
            ExecutionProvider::CUDA => builder.with_cuda(0).map_err(|e| crate::error::InferaError::OnnxRuntimeError(e.to_string()))?,
            _ => return Err(crate::error::InferaError::ExecutionProviderNotAvailable(provider.to_string())),
        };

        builder
            .with_model_from_file(model_path)
            .map_err(|e| crate::error::InferaError::OnnxRuntimeError(e.to_string()))
    }
}

/// A global, thread-safe store for all loaded ONNX models.
///
/// This is a `Lazy` static, meaning it is initialized on first access.
/// It uses a `RwLock` to allow multiple concurrent reads and exclusive writes,
/// mapping model names (strings) to their `OnnxModel` representations.
pub(crate) static MODELS: Lazy<RwLock<HashMap<String, OnnxModel>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
