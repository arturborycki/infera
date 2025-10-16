// src/engine.rs
// Contains the core ONNX inference logic using the Tract library.

use crate::error::InferaError;
use crate::ffi_utils::InferaInferenceResult;
use crate::model::{OnnxModel, MODELS, ModelType};
use serde_json::json;
use std::convert::TryInto;
use std::mem;

#[cfg(feature = "tract")]
use tract_onnx::prelude::*;

/// Loads, compiles, and stores an ONNX model.
///
/// This function reads an ONNX model from the given path, uses the Tract library
/// to parse, optimize, and compile it into a runnable plan. It also extracts
/// metadata like input/output shapes. The resulting `OnnxModel` is then
/// inserted into the global `MODELS` map.
///
/// # Arguments
///
/// * `name` - The name to assign to the loaded model.
/// * `path` - The file system path to the `.onnx` model file.
///
/// # Returns
///
/// * `Ok(())` on successful loading and compilation.
/// * `Err(InferaError)` if the model cannot be found, parsed, or compiled.
#[cfg(feature = "tract")]
pub(crate) fn load_model_impl(name: &str, path: &str) -> Result<(), InferaError> {
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?
        .into_optimized()
        .map_err(|e| InferaError::OnnxError(e.to_string()))?
        .into_runnable()
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let input_facts = model
        .model()
        .input_fact(0)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let output_facts = model
        .model()
        .output_fact(0)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let input_shape: Vec<i64> = input_facts
        .shape
        .iter()
        .map(|d| d.to_i64().unwrap_or(-1))
        .collect();
    let output_shape: Vec<i64> = output_facts
        .shape
        .iter()
        .map(|d| d.to_i64().unwrap_or(-1))
        .collect();
    let onnx_model = OnnxModel {
        model,
        input_shape,
        output_shape,
        name: name.to_string(),
        model_type: ModelType::Numerical,  // Default to numerical model
    };
    MODELS.write().insert(name.to_string(), onnx_model);
    Ok(())
}

/// A stub for `load_model_impl` when the "tract" feature is disabled.
///
/// Always returns an `InferaError::FeatureNotEnabled` error.
#[cfg(not(feature = "tract"))]
pub(crate) fn load_model_impl(_name: &str, _path: &str) -> Result<(), InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

/// Loads, compiles, and stores an ONNX model configured for text processing.
///
/// Similar to `load_model_impl` but configures the model for text processing
/// with the specified tokenizer and maximum sequence length.
///
/// # Arguments
///
/// * `name` - The name to assign to the loaded model.
/// * `path` - The file system path to the `.onnx` model file.
/// * `tokenizer_path` - The path to the tokenizer JSON file.
/// * `max_length` - The maximum sequence length for tokenization.
///
/// # Returns
///
/// * `Ok(())` on successful loading and compilation.
/// * `Err(InferaError)` if loading, optimization, or compilation fails.
#[cfg(feature = "tract")]
pub(crate) fn load_text_model_impl(
    name: &str,
    path: &str,
    tokenizer_path: &str,
    max_length: usize,
) -> Result<(), InferaError> {
    use crate::text::TextModelConfig;
    
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?
        .into_optimized()
        .map_err(|e| InferaError::OnnxError(e.to_string()))?
        .into_runnable()
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let input_facts = model
        .model()
        .input_fact(0)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let output_facts = model
        .model()
        .output_fact(0)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let input_shape: Vec<i64> = input_facts
        .shape
        .iter()
        .map(|d| d.to_i64().unwrap_or(-1))
        .collect();
    let output_shape: Vec<i64> = output_facts
        .shape
        .iter()
        .map(|d| d.to_i64().unwrap_or(-1))
        .collect();
    
    let text_config = TextModelConfig {
        tokenizer_path: tokenizer_path.to_string(),
        max_length,
    };
    
    let onnx_model = OnnxModel {
        model,
        input_shape,
        output_shape,
        name: name.to_string(),
        model_type: ModelType::Text(text_config),
    };
    MODELS.write().insert(name.to_string(), onnx_model);
    Ok(())
}

/// A stub for `load_text_model_impl` when the "tract" feature is disabled.
#[cfg(not(feature = "tract"))]
pub(crate) fn load_text_model_impl(
    _name: &str,
    _path: &str,
    _tokenizer_path: &str,
    _max_length: usize,
) -> Result<(), InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

/// Runs inference with a given model and input tensor data.
///
/// This function retrieves the specified model from the global `MODELS` store,
/// constructs a Tract tensor from the raw input data, runs the model,
/// and packages the output into an `InferaInferenceResult`.
///
/// # Arguments
///
/// * `model_name` - The name of the loaded model to use for inference.
/// * `data` - A pointer to the raw f32 tensor data.
/// * `rows` - The number of rows in the input tensor.
/// * `cols` - The number of columns in the input tensor.
///
/// # Returns
///
/// * `Ok(InferaInferenceResult)` containing the output tensor data and metadata.
/// * `Err(InferaError)` if the model is not found or if an error occurs during inference.
#[cfg(feature = "tract")]
pub(crate) fn run_inference_impl(
    model_name: &str,
    data: *const f32,
    rows: usize,
    cols: usize,
) -> Result<InferaInferenceResult, InferaError> {
    let models = MODELS.read();
    let model = models
        .get(model_name)
        .ok_or_else(|| InferaError::ModelNotFound(model_name.to_string()))?;
    let input_data = unsafe { std::slice::from_raw_parts(data, rows * cols) };
    let input_tensor = Tensor::from_shape(&[rows, cols], input_data)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let outputs = model
        .model
        .run(tvec!(input_tensor.into()))
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let output_tensor = outputs
        .into_iter()
        .next()
        .ok_or_else(|| InferaError::OnnxError("No output tensor".to_string()))?;
    let output_array = output_tensor
        .to_array_view::<f32>()
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let output_shape = output_array.shape();
    let output_rows = output_shape[0];
    let output_cols = if output_shape.len() > 1 {
        output_shape[1]
    } else {
        1
    };
    let output_data: Vec<f32> = output_array.iter().cloned().collect();
    let output_len = output_data.len();
    let output_ptr = Box::into_raw(output_data.into_boxed_slice()) as *mut f32;
    Ok(InferaInferenceResult {
        data: output_ptr,
        len: output_len,
        rows: output_rows,
        cols: output_cols,
        status: 0,
    })
}

/// A stub for `run_inference_impl` when the "tract" feature is disabled.
///
/// Always returns an `InferaError::FeatureNotEnabled` error.
#[cfg(not(feature = "tract"))]
pub(crate) fn run_inference_impl(
    _model_name: &str,
    _data: *const f32,
    _rows: usize,
    _cols: usize,
) -> Result<InferaInferenceResult, InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

/// Runs inference with a given model and raw BLOB input data.
///
/// This function is similar to `run_inference_impl` but takes a raw byte slice (`BLOB`)
/// as input. It converts the bytes to `f32` values, validates the shape against the
/// model's expected input, and attempts to infer the batch size for models with
/// dynamic input dimensions.
///
/// # Arguments
///
/// * `model_name` - The name of the loaded model to use for inference.
/// * `blob_data` - A pointer to the raw byte data.
/// * `blob_len` - The length of the byte data slice.
///
/// # Returns
///
/// * `Ok(InferaInferenceResult)` containing the output tensor data and metadata.
/// * `Err(InferaError)` if the model is not found, the blob size is invalid,
///   the shape does not match, or an error occurs during inference.
#[cfg(feature = "tract")]
pub(crate) fn run_inference_blob_impl(
    model_name: &str,
    blob_data: *const u8,
    blob_len: usize,
) -> Result<InferaInferenceResult, InferaError> {
    let models = MODELS.read();
    let model = models
        .get(model_name)
        .ok_or_else(|| InferaError::ModelNotFound(model_name.to_string()))?;
    if !blob_len.is_multiple_of(mem::size_of::<f32>()) {
        return Err(InferaError::InvalidBlobSize);
    }
    let blob_bytes = unsafe { std::slice::from_raw_parts(blob_data, blob_len) };
    let float_vec: Vec<f32> = blob_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();
    let expected_elements: usize = model
        .input_shape
        .iter()
        .filter(|&&d| d > 0)
        .map(|&d| d as usize)
        .product();
    if !float_vec.len().is_multiple_of(expected_elements) {
        return Err(InferaError::BlobShapeMismatch {
            expected: expected_elements,
            actual: float_vec.len(),
        });
    }
    let batch_size = float_vec.len() / expected_elements;
    let final_shape: Vec<usize> = model
        .input_shape
        .iter()
        .map(|&d| if d == -1 { batch_size } else { d as usize })
        .collect();
    let input_tensor = Tensor::from_shape(&final_shape, &float_vec)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let outputs = model
        .model
        .run(tvec!(input_tensor.into()))
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let output_tensor = outputs
        .into_iter()
        .next()
        .ok_or_else(|| InferaError::OnnxError("No output tensor".to_string()))?;
    let output_array = output_tensor
        .to_array_view::<f32>()
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let output_shape = output_array.shape();
    let output_rows = output_shape[0];
    let output_cols = if output_shape.len() > 1 {
        output_shape[1]
    } else {
        1
    };
    let output_data: Vec<f32> = output_array.iter().cloned().collect();
    let output_len = output_data.len();
    let output_ptr = Box::into_raw(output_data.into_boxed_slice()) as *mut f32;
    Ok(InferaInferenceResult {
        data: output_ptr,
        len: output_len,
        rows: output_rows,
        cols: output_cols,
        status: 0,
    })
}

/// A stub for `run_inference_blob_impl` when the "tract" feature is disabled.
///
/// Always returns an `InferaError::FeatureNotEnabled` error.
#[cfg(not(feature = "tract"))]
pub(crate) fn run_inference_blob_impl(
    _model_name: &str,
    _blob_data: *const u8,
    _blob_len: usize,
) -> Result<InferaInferenceResult, InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

/// Retrieves metadata for a loaded model as a JSON string.
///
/// This function looks up the model by name and serializes its metadata
/// (name, input shape, output shape) into a JSON string.
///
/// # Arguments
///
/// * `model_name` - The name of the loaded model.
///
/// # Returns
///
/// * `Ok(String)` containing the JSON metadata.
/// * `Err(InferaError)` if the model is not found or if JSON serialization fails.
#[cfg(feature = "tract")]
pub(crate) fn get_model_metadata_impl(model_name: &str) -> Result<String, InferaError> {
    let models = MODELS.read();
    let model = models
        .get(model_name)
        .ok_or_else(|| InferaError::ModelNotFound(model_name.to_string()))?;
    let info = json!({
        "name": model.name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "loaded": true
    });
    serde_json::to_string(&info).map_err(|e| InferaError::JsonError(e.to_string()))
}

/// A stub for `get_model_metadata_impl` when the "tract" feature is disabled.
///
/// Always returns an `InferaError::FeatureNotEnabled` error.
#[cfg(not(feature = "tract"))]
pub(crate) fn get_model_metadata_impl(_model_name: &str) -> Result<String, InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "ONNX inference requires 'tract' feature to be enabled".to_string(),
    ))
}

/// Runs text inference on a loaded model.
///
/// This function takes a text input, processes it using the model's configured
/// tokenizer (if it's a text model), and runs inference to generate embeddings
/// or other text-based outputs.
///
/// # Arguments
///
/// * `model_name` - The name of the loaded model.
/// * `text` - The input text to process.
///
/// # Returns
///
/// * `Ok(InferaInferenceResult)` containing the inference output.
/// * `Err(InferaError)` if the model is not found, not a text model, or inference fails.
#[cfg(all(feature = "tract", feature = "text"))]
pub(crate) fn run_text_inference_impl(
    model_name: &str,
    text: &str,
) -> Result<InferaInferenceResult, InferaError> {
    use crate::text::TextProcessor;
    
    let models = MODELS.read();
    let model = models
        .get(model_name)
        .ok_or_else(|| InferaError::ModelNotFound(model_name.to_string()))?;

    // Check if this is a text model
    let text_config = match &model.model_type {
        ModelType::Text(config) => config,
        ModelType::Numerical => {
            return Err(InferaError::TextProcessingError(
                format!("Model '{}' is not configured for text processing", model_name)
            ));
        }
    };

    // Create text processor and encode the text
    let processor = TextProcessor::new(&text_config.tokenizer_path, text_config.max_length)?;
    let (input_ids, attention_mask) = processor.encode_text(text)?;

    // Create token_type_ids (all zeros for single sentence)
    let token_type_ids: Vec<i64> = vec![0; input_ids.len()];

    // Create tensors for all three inputs as i64 (integers)
    let input_ids_tensor = Tensor::from_shape(&[1, input_ids.len()], &input_ids)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let attention_mask_tensor = Tensor::from_shape(&[1, attention_mask.len()], &attention_mask)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;
    let token_type_ids_tensor = Tensor::from_shape(&[1, token_type_ids.len()], &token_type_ids)
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;

    let outputs = model
        .model
        .run(tvec!(input_ids_tensor.into(), attention_mask_tensor.into(), token_type_ids_tensor.into()))
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;

    let output_tensor = outputs
        .into_iter()
        .next()
        .ok_or_else(|| InferaError::OnnxError("No output tensor".to_string()))?;

    let output_array = output_tensor
        .to_array_view::<f32>()
        .map_err(|e| InferaError::OnnxError(e.to_string()))?;

    // The output shape is [batch_size, seq_length, hidden_size] = [1, 512, 768]
    let output_shape = output_array.shape();
    if output_shape.len() != 3 {
        return Err(InferaError::OnnxError(format!("Expected 3D output tensor, got {:?}", output_shape)));
    }
    
    let batch_size = output_shape[0];
    let seq_length = output_shape[1];
    let hidden_size = output_shape[2];
    
    if batch_size != 1 {
        return Err(InferaError::OnnxError(format!("Expected batch size 1, got {}", batch_size)));
    }

    // Perform mean pooling over the sequence dimension
    // Only consider non-padding tokens (where attention_mask == 1)
    let mut pooled_embeddings = vec![0.0f32; hidden_size];
    let mut total_tokens = 0;

    for seq_idx in 0..seq_length {
        if attention_mask[seq_idx] == 1 {
            for hidden_idx in 0..hidden_size {
                pooled_embeddings[hidden_idx] += output_array[[0, seq_idx, hidden_idx]];
            }
            total_tokens += 1;
        }
    }

    // Average the embeddings
    if total_tokens > 0 {
        for embedding in pooled_embeddings.iter_mut() {
            *embedding /= total_tokens as f32;
        }
    }

    // Convert to the expected result format
    let len = pooled_embeddings.len();
    let raw_ptr = Box::<[f32]>::into_raw(pooled_embeddings.into_boxed_slice());
    let data_ptr = unsafe { (*raw_ptr).as_mut_ptr() };
    let _ = raw_ptr;

    Ok(InferaInferenceResult {
        data: data_ptr,
        len,
        rows: 1,
        cols: len,
        status: 0,
    })
}

/// Stub for `run_text_inference_impl` when features are disabled.
#[cfg(not(all(feature = "tract", feature = "text")))]
pub(crate) fn run_text_inference_impl(
    _model_name: &str,
    _text: &str,
) -> Result<InferaInferenceResult, InferaError> {
    Err(InferaError::FeatureNotEnabled(
        "Text inference requires both 'tract' and 'text' features to be enabled".to_string(),
    ))
}
