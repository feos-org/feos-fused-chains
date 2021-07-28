use crate::FusedChainFunctional;
use ::quantity::pyquantity::*;
use eos_core::python::{PyContributions, PyVerbosity};
use eos_core::*;
use eos_dft::adsorption::*;
use eos_dft::python::*;
use eos_dft::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use quantity::si::*;
use std::rc::Rc;

/// Helmholtz energy functional for fused chains.
#[pyclass(name = "FusedChainFunctional", unsendable)]
#[derive(Clone)]
pub struct PyFusedChainFunctional {
    pub _data: Rc<DFT<FusedChainFunctional>>,
}

#[pymethods]
impl PyFusedChainFunctional {
    /// New functional for fused dimers.
    ///
    /// Parameters
    /// ----------
    /// sigma1: float
    ///     Diameter of the first segment.
    /// sigma2: float
    ///     Diameter of the second segment.
    /// distance: float
    ///     Distance between the centers of the two segments.
    ///
    /// Returns
    /// -------
    /// FusedChainFunctional
    #[staticmethod]
    #[pyo3(text_signature = "(sigma1, sigma2, distance)")]
    fn new_dimer(sigma1: f64, sigma2: f64, distance: f64) -> Self {
        Self {
            _data: Rc::new(FusedChainFunctional::new_dimer(sigma1, sigma2, distance)),
        }
    }

    /// New functional for monomers.
    ///
    /// Parameters
    /// ----------
    /// sigma: float
    ///     Diameter of the monomer.
    ///
    /// Returns
    /// -------
    /// FusedChainFunctional
    #[staticmethod]
    #[pyo3(text_signature = "(sigma)")]
    fn new_monomer(sigma: f64) -> Self {
        Self {
            _data: Rc::new(FusedChainFunctional::new_monomer(sigma)),
        }
    }
}

impl_state!(DFT<FusedChainFunctional>, PyFusedChainFunctional);

impl_adsorption_profile!(FusedChainFunctional);

impl_adsorption!(FusedChainFunctional, PyFusedChainFunctional);

#[pymodule]
pub fn fused_chain(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(quantity))?;
    m.add_class::<PyFusedChainFunctional>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyAdsorptionProfile>()?;
    m.add_class::<PyExternalPotential>()?;
    m.add_class::<PyAdsorption>()?;
    m.add_class::<PyDFTSolver>()?;

    py.run(
        "import sys; sys.modules['fused_chain.si'] = quantity",
        None,
        Some(m.dict()),
    )
}
