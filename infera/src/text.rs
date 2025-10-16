// src/text.rs
// Text processing module for handling tokenization and text-to-tensor conversion

use crate::error::InferaError;

#[cfg(feature = "text")]
use tokenizers::Tokenizer;

/// Configuration for text models
#[derive(Debug, Clone)]
pub struct TextModelConfig {
    pub tokenizer_path: String,
    pub max_length: usize,
}

/// Text processor for handling tokenization
#[cfg(feature = "text")]
pub struct TextProcessor {
    tokenizer: Tokenizer,
    max_length: usize,
}

#[cfg(feature = "text")]
impl TextProcessor {
    /// Create a new text processor with the given tokenizer
    pub fn new(tokenizer_path: &str, max_length: usize) -> Result<Self, InferaError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| InferaError::TextProcessingError(format!("Failed to load tokenizer: {}", e)))?;
        
        Ok(Self {
            tokenizer,
            max_length,
        })
    }

    /// Encode text to token IDs and attention mask
    pub fn encode_text(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>), InferaError> {
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| InferaError::TextProcessingError(e.to_string()))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mut attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&mask| mask as i64).collect();

        // Truncate if longer than max_length
        if input_ids.len() > self.max_length {
            input_ids.truncate(self.max_length);
            attention_mask.truncate(self.max_length);
        }

        // Pad to max_length instead of fixed 768
        while input_ids.len() < self.max_length {
            input_ids.push(0); // PAD token
            attention_mask.push(0); // No attention for padding
        }

        Ok((input_ids, attention_mask))
    }

    /// Convert text to float tensor (for simple models that expect float input)
    pub fn text_to_float_tensor(&self, text: &str) -> Result<Vec<f32>, InferaError> {
        let (input_ids, _) = self.encode_text(text)?;
        Ok(input_ids.iter().map(|&id| id as f32).collect())
    }
}

/// Stub implementation when text feature is disabled
#[cfg(not(feature = "text"))]
pub struct TextProcessor;

#[cfg(not(feature = "text"))]
impl TextProcessor {
    pub fn new(_tokenizer_path: &str, _max_length: usize) -> Result<Self, InferaError> {
        Err(InferaError::FeatureNotEnabled(
            "Text processing requires 'text' feature to be enabled".to_string(),
        ))
    }

    pub fn encode_text(&self, _text: &str) -> Result<(Vec<i64>, Vec<i64>), InferaError> {
        Err(InferaError::FeatureNotEnabled(
            "Text processing requires 'text' feature to be enabled".to_string(),
        ))
    }

    pub fn text_to_float_tensor(&self, _text: &str) -> Result<Vec<f32>, InferaError> {
        Err(InferaError::FeatureNotEnabled(
            "Text processing requires 'text' feature to be enabled".to_string(),
        ))
    }
}