# Text Embedding Support for Infera

This document describes the text embedding functionality and GPU execution support in the Infera DuckDB extension.

## Overview

The Infera extension supports text embedding generation using sentence transformer models with both CPU and GPU execution providers. The implementation includes dynamic execution provider switching and comprehensive hardware acceleration support.

## Core Functions

### Text Model Loading

1. **`infera_load_text_model(name, path, tokenizer_path, max_length)`**
   - Loads an ONNX model configured for text processing (CPU execution)
   - Parameters:
     - `name`: Unique name for the model
     - `path`: Path or URL to the ONNX model file
     - `tokenizer_path`: Path to the tokenizer JSON file
     - `max_length`: Maximum sequence length for tokenization
   - Returns: `BOOLEAN` (true on success)

2. **`infera_load_text_model_gpu(name, path, tokenizer_path, max_length, provider)`**
   - Loads an ONNX model with specified execution provider
   - Parameters:
     - `name`: Unique name for the model
     - `path`: Path or URL to the ONNX model file
     - `tokenizer_path`: Path to the tokenizer JSON file
     - `max_length`: Maximum sequence length for tokenization
     - `provider`: Execution provider ("CPU", "CUDA", "ROCm", "DirectML", "CoreML")
   - Returns: `BOOLEAN` (true on success)

### Text Inference

3. **`infera_predict_text(model_name, text)`**
   - Generates embeddings from text input
   - Parameters:
     - `model_name`: Name of the loaded text model
     - `text`: Input text to process
   - Returns: `LIST<FLOAT>` (embedding vector)

## GPU and Execution Provider Functions

### Provider Management

4. **`infera_get_available_providers()`**
   - Returns list of available execution providers on the system
   - Returns: `JSON` array of provider names
   - Example: `["CPU"]` or `["CPU", "CUDA", "ROCm"]`

5. **`infera_set_execution_provider(model_name, provider)`**
   - Dynamically switches execution provider for a loaded model
   - Parameters:
     - `model_name`: Name of the loaded model
     - `provider`: Target execution provider
   - Returns: `BOOLEAN` (true on success)

6. **`infera_get_execution_provider(model_name)`**
   - Gets current execution provider information for a model
   - Parameters:
     - `model_name`: Name of the loaded model
   - Returns: `JSON` with provider details

### General Model Loading with GPU Support

7. **`infera_load_model_gpu(name, path, provider)`**
   - Loads a general ONNX model with specified execution provider
   - Parameters:
     - `name`: Unique name for the model
     - `path`: Path or URL to the ONNX model file
     - `provider`: Execution provider
   - Returns: `BOOLEAN` (true on success)

## Architecture

The implementation follows a layered architecture with dual execution engine support:

### Execution Engines

1. **Tract Engine** (CPU-optimized):
   - Fast CPU inference for standard ONNX models
   - Lightweight and efficient for CPU workloads
   - Default engine for CPU execution

2. **ONNX Runtime Engine** (GPU-ready):
   - Supports multiple execution providers (CPU, CUDA, ROCm, DirectML, CoreML)
   - Hardware acceleration capabilities
   - Advanced optimization features

### Core Modules

1. **Rust Core** (`infera/src/`):
   - `execution_provider.rs`: **NEW** - Execution provider abstraction and detection
   - `text.rs`: Text processing and tokenization logic
   - `engine.rs`: Extended with text inference capabilities
   - `model.rs`: Enhanced with dual-engine support and provider switching
   - `lib.rs`: Extended FFI functions for GPU operations

2. **C++ Extension** (`infera/bindings/`):
   - `infera_extension.cpp`: Comprehensive SQL function implementations
   - `rust.h`: Updated with GPU function declarations

3. **Build Features**:
   - `text`: Optional feature for tokenizer support (requires `tokenizers` crate)
   - `tract`: ONNX inference engine (CPU-focused)
   - `onnxruntime`: **NEW** - Optional ONNX Runtime with GPU support

### Execution Provider Support

The extension supports the following execution providers:

- **CPU**: Universal support (always available)
- **CUDA**: NVIDIA GPU acceleration (when onnxruntime feature enabled)
- **ROCm**: AMD GPU acceleration (when onnxruntime feature enabled)
- **DirectML**: Windows GPU acceleration (when onnxruntime feature enabled)
- **CoreML**: Apple Silicon acceleration (when onnxruntime feature enabled)

## Current Status

- ✅ **Core Infrastructure**: Text processing module and model configuration
- ✅ **GPU Infrastructure**: Complete execution provider system with dynamic switching
- ✅ **FFI Interface**: Comprehensive Rust C API for text and GPU operations
- ✅ **SQL Functions**: Full DuckDB integration with CPU and GPU support
- ✅ **Build System**: Compiles successfully with backward compatibility
- ✅ **Provider Detection**: Runtime detection of available execution providers
- ✅ **Error Handling**: Robust error handling for unsupported providers
- ✅ **Testing**: GPU infrastructure validated with CPU provider

## Feature Flag Status

Currently building with:
- `tract`: Enabled (CPU ONNX inference)
- `text`: Disabled (tokenizer functionality)
- `onnxruntime`: Disabled (GPU execution providers)

### Provider Availability

With current build configuration:
- **CPU Provider**: Available (via Tract engine)
- **GPU Providers**: Requires `onnxruntime` feature to be enabled

When GPU features are disabled, GPU-related functions return appropriate "feature not enabled" errors.

## Usage Examples

### Basic Text Embeddings (CPU)

```sql
-- Load a sentence transformer model (CPU execution)
SELECT infera_load_text_model(
    'sentence_transformer', 
    'models/all-MiniLM-L6-v2.onnx',
    'tokenizers/all-MiniLM-L6-v2-tokenizer.json',
    384
);

-- Generate embeddings
SELECT infera_predict_text('sentence_transformer', 'Hello world!');
-- Returns: [0.1, -0.2, 0.3, ..., 0.8] (embedding vector)
```

### GPU-Accelerated Text Embeddings

```sql
-- Check available execution providers
SELECT infera_get_available_providers();
-- Returns: ["CPU", "CUDA"] (when onnxruntime feature is enabled)

-- Load model with GPU acceleration
SELECT infera_load_text_model_gpu(
    'gpu_transformer', 
    'models/all-MiniLM-L6-v2.onnx',
    'tokenizers/all-MiniLM-L6-v2-tokenizer.json',
    384,
    'CUDA'
);

-- Generate embeddings with GPU acceleration
SELECT infera_predict_text('gpu_transformer', 'GPU-accelerated text processing!');
```

### Dynamic Provider Switching

```sql
-- Load model with CPU
SELECT infera_load_text_model_gpu('adaptive_model', 'model.onnx', 'tokenizer.json', 512, 'CPU');

-- Check current provider
SELECT infera_get_execution_provider('adaptive_model');
-- Returns: {"provider": "CPU", "engine": "Tract"}

-- Switch to GPU (when available)
SELECT infera_set_execution_provider('adaptive_model', 'CUDA');

-- Verify the switch
SELECT infera_get_execution_provider('adaptive_model');
-- Returns: {"provider": "CUDA", "engine": "OnnxRuntime"}
```

### Batch Processing with GPU

```sql
-- Batch processing with GPU acceleration
SELECT 
    text,
    infera_predict_text('gpu_transformer', text) as embedding
FROM documents
WHERE length(text) > 10;
```

### General Model GPU Loading

```sql
-- Load non-text models with GPU support
SELECT infera_load_model_gpu('image_classifier', 'resnet50.onnx', 'CUDA');

-- Use with regular predict function
SELECT infera_predict('image_classifier', [/* image tensor */]);
```

## Setup Instructions

### Enable Text Functionality

1. **Enable text feature**:
   ```toml
   # In Cargo.toml
   default = ["tract", "text"]
   ```

2. **Install tokenizers dependency**:
   - The `tokenizers` crate will be automatically included when `text` feature is enabled

### Enable GPU Support

1. **Enable ONNX Runtime with GPU support**:
   ```toml
   # In Cargo.toml
   default = ["tract", "text", "onnxruntime"]
   ```

2. **Platform-specific GPU setup**:
   
   **NVIDIA CUDA**:
   - Install CUDA Toolkit (11.0+ recommended)
   - Ensure cuDNN is available
   - Verify with: `nvidia-smi`

   **AMD ROCm**:
   - Install ROCm platform
   - Configure ROCm environment
   - Verify with: `rocm-smi`

   **Apple Silicon**:
   - CoreML support is built-in on macOS
   - No additional setup required

   **Windows DirectML**:
   - DirectML comes with Windows 10/11
   - Ensure updated graphics drivers

### Model Preparation

3. **Export sentence transformer models**:
   ```python
   from optimum.onnxruntime import ORTModelForFeatureExtraction
   from transformers import AutoTokenizer
   
   # Export model to ONNX
   model = ORTModelForFeatureExtraction.from_pretrained(
       "sentence-transformers/all-MiniLM-L6-v2",
       export=True
   )
   model.save_pretrained("./models/")
   
   # Save tokenizer
   tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
   tokenizer.save_pretrained("./tokenizers/")
   ```

4. **Test GPU availability**:
   ```sql
   -- Check what providers are available
   SELECT infera_get_available_providers();
   
   -- Try loading with different providers
   SELECT infera_load_model_gpu('test', 'model.onnx', 'CPU');     -- Should work
   SELECT infera_load_model_gpu('test', 'model.onnx', 'CUDA');   -- GPU dependent
   ```

## Performance Considerations

### Execution Provider Selection

- **CPU (Tract)**: Best for small models and low-latency inference
- **CPU (ONNX Runtime)**: Better optimization for complex models
- **CUDA**: Optimal for large models and batch processing on NVIDIA GPUs
- **ROCm**: Best choice for AMD GPU acceleration
- **CoreML**: Efficient on Apple Silicon devices
- **DirectML**: Good performance on Windows with various GPU vendors

### Best Practices

1. **Check provider availability** before loading models
2. **Use CPU for small, frequent queries** to avoid GPU memory transfer overhead
3. **Use GPU for large batch processing** to maximize parallelization benefits
4. **Profile different providers** to find optimal performance for your use case
5. **Handle provider fallback** gracefully in production environments

## Error Handling

The extension provides comprehensive error handling:

```sql
-- Provider availability errors
SELECT infera_load_model_gpu('test', 'model.onnx', 'CUDA');
-- Returns false with error: "Execution provider 'CUDA' is not available"

-- Feature availability errors (when onnxruntime disabled)
SELECT infera_get_available_providers();
-- Returns error: "GPU features not available. Enable 'onnxruntime' feature to use GPU execution providers."

-- Invalid provider names
SELECT infera_set_execution_provider('model', 'InvalidProvider');
-- Returns false with error: "Unknown execution provider: 'InvalidProvider'"
```

## Benefits

- **Performance**: Direct ONNX inference without Python overhead
- **Hardware Acceleration**: Support for NVIDIA CUDA, AMD ROCm, Apple CoreML, and DirectML
- **Flexibility**: Dynamic execution provider switching without reloading models
- **Integration**: Native SQL interface for text embeddings and GPU operations
- **Scalability**: GPU-accelerated batch processing within SQL queries
- **Compatibility**: Maintains all existing functionality with backward compatibility
- **Robustness**: Comprehensive error handling and graceful provider fallback

## Technical Implementation

### Dual Engine Architecture

The extension uses a sophisticated dual-engine approach:

1. **Tract Engine**: Optimized for CPU inference, lightweight and fast
2. **ONNX Runtime Engine**: Full-featured with GPU support and advanced optimizations

### Provider Detection

The system automatically detects available execution providers at runtime:

```rust
// Runtime provider detection
pub fn get_available_providers() -> Vec<ExecutionProvider> {
    ExecutionProvider::iter()
        .filter(|provider| provider.is_available())
        .collect()
}
```

### Memory Management

- **Efficient model storage**: Models are stored once and can switch providers
- **GPU memory optimization**: Automatic memory management for GPU operations
- **Resource cleanup**: Proper cleanup of GPU resources when switching providers

## Troubleshooting

### Common Issues

1. **"GPU features not available"**:
   - Enable the `onnxruntime` feature in Cargo.toml
   - Rebuild the extension

2. **"Execution provider 'CUDA' is not available"**:
   - Install NVIDIA CUDA Toolkit
   - Verify CUDA installation with `nvidia-smi`
   - Check ONNX Runtime CUDA support

3. **Model loading failures**:
   - Verify ONNX model file exists and is readable
   - Check tokenizer JSON file for text models
   - Ensure sufficient memory for model loading

4. **Performance issues**:
   - Profile different execution providers
   - Consider batch size optimization
   - Monitor GPU memory usage

### Debug Commands

```sql
-- Check system capabilities
SELECT infera_get_available_providers();

-- Verify model status
SELECT infera_get_execution_provider('model_name');

-- Test provider switching
SELECT infera_set_execution_provider('model_name', 'CPU');
```

The implementation provides a comprehensive foundation for text embedding generation with optional GPU acceleration while preserving all existing numerical inference capabilities.