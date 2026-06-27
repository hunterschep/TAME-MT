mod fingerprint;
mod index;
mod ngrams;
mod similarity;
mod types;
mod validation;

use index::NativeNgramIndex;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyfunction]
fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn native_thread_count() -> usize {
    rayon::current_num_threads()
}

#[pyfunction]
fn configure_native_threads(threads: Option<usize>) -> PyResult<usize> {
    let Some(requested_threads) = threads else {
        return Ok(rayon::current_num_threads());
    };
    if requested_threads == 0 {
        return Err(PyValueError::new_err(
            "native thread count must be a positive integer",
        ));
    }
    match rayon::ThreadPoolBuilder::new()
        .num_threads(requested_threads)
        .build_global()
    {
        Ok(()) => Ok(rayon::current_num_threads()),
        Err(_) => {
            let active_threads = rayon::current_num_threads();
            if active_threads == requested_threads {
                Ok(active_threads)
            } else {
                Err(PyRuntimeError::new_err(format!(
                    "native thread pool is already initialized with {active_threads} threads; \
                     requested {requested_threads}"
                )))
            }
        }
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeNgramIndex>()?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(native_thread_count, m)?)?;
    m.add_function(wrap_pyfunction!(configure_native_threads, m)?)?;
    Ok(())
}
