mod index;
mod ngrams;
mod similarity;
mod types;
mod validation;

use index::NativeNgramIndex;
use pyo3::prelude::*;

#[pyfunction]
fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn native_thread_count() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeNgramIndex>()?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(native_thread_count, m)?)?;
    Ok(())
}
