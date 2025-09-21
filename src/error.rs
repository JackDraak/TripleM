use std::fmt;

/// Result type alias for the mood music module
pub type Result<T> = std::result::Result<T, MoodMusicError>;

/// Error types for the mood music module
#[derive(Debug, Clone, PartialEq)]
pub enum MoodMusicError {
    /// Audio system initialization failed
    AudioInitializationFailed(String),

    /// Invalid configuration parameter
    InvalidConfiguration(String),

    /// Audio stream error
    AudioStreamError(String),

    /// Buffer underrun or overrun
    BufferError(String),

    /// Generator initialization failed
    GeneratorError(String),

    /// Pattern generation failed
    PatternError(String),

    /// Voice coordination error
    VoiceCoordinationError(String),

    /// Generic internal error
    InternalError(String),
}

impl fmt::Display for MoodMusicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MoodMusicError::AudioInitializationFailed(msg) => {
                write!(f, "Audio initialization failed: {}", msg)
            }
            MoodMusicError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            MoodMusicError::AudioStreamError(msg) => {
                write!(f, "Audio stream error: {}", msg)
            }
            MoodMusicError::BufferError(msg) => {
                write!(f, "Buffer error: {}", msg)
            }
            MoodMusicError::GeneratorError(msg) => {
                write!(f, "Generator error: {}", msg)
            }
            MoodMusicError::PatternError(msg) => {
                write!(f, "Pattern error: {}", msg)
            }
            MoodMusicError::VoiceCoordinationError(msg) => {
                write!(f, "Voice coordination error: {}", msg)
            }
            MoodMusicError::InternalError(msg) => {
                write!(f, "Internal error: {}", msg)
            }
        }
    }
}

impl std::error::Error for MoodMusicError {}

impl From<cpal::BuildStreamError> for MoodMusicError {
    fn from(err: cpal::BuildStreamError) -> Self {
        MoodMusicError::AudioStreamError(err.to_string())
    }
}

impl From<cpal::PlayStreamError> for MoodMusicError {
    fn from(err: cpal::PlayStreamError) -> Self {
        MoodMusicError::AudioStreamError(err.to_string())
    }
}

impl From<cpal::PauseStreamError> for MoodMusicError {
    fn from(err: cpal::PauseStreamError) -> Self {
        MoodMusicError::AudioStreamError(err.to_string())
    }
}