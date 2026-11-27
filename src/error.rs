//! Error compartido por todo el crate.

use std::fmt;

#[derive(Debug, Clone)]
pub struct Error {
    message: String,
}

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::new(error.to_string())
    }
}

impl From<std::string::FromUtf8Error> for Error {
    fn from(error: std::string::FromUtf8Error) -> Self {
        Self::new(error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Azúcar para construir errores con formato, equivalente a `throw std::invalid_argument`.
#[macro_export]
macro_rules! bail {
    ($($argument:tt)*) => {
        return Err($crate::error::Error::new(format!($($argument)*)))
    };
}

#[macro_export]
macro_rules! ensure {
    ($condition:expr, $($argument:tt)*) => {
        if !($condition) {
            return Err($crate::error::Error::new(format!($($argument)*)));
        }
    };
}
