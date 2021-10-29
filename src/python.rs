use crate::FusedChainFunctional;
use ::quantity::python::*;
use eos_dft::adsorption::*;
use eos_dft::fundamental_measure_theory::FMTVersion;
use eos_dft::python::*;
use eos_dft::*;
use feos_core::python::{PyContributions, PyVerbosity};
use feos_core::*;
use numpy::{PyArray1, PyArray2, PyArray4, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use quantity::si::*;
use std::rc::Rc;

/// Helmholtz energy functional for fused chains.
#[pyclass(name = "FusedChainFunctional", unsendable)]
#[derive(Clone)]
pub struct PyFusedChainFunctional(Rc<DFT<FusedChainFunctional>>);

#[pymethods]
impl PyFusedChainFunctional {
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
    fn new_monomer(sigma: f64, kierlik_rosinberg: Option<bool>) -> Self {
        let mut version = FMTVersion::WhiteBear;
        if let Some(kierlik_rosinberg) = kierlik_rosinberg {
            if kierlik_rosinberg {
                version = FMTVersion::KierlikRosinberg;
            }
        }
        Self(Rc::new(FusedChainFunctional::new_monomer(sigma, version)))
    }

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
    fn new_dimer(sigma1: f64, sigma2: f64, distance: f64, kierlik_rosinberg: Option<bool>) -> Self {
        let mut version = FMTVersion::WhiteBear;
        if let Some(kierlik_rosinberg) = kierlik_rosinberg {
            if kierlik_rosinberg {
                version = FMTVersion::KierlikRosinberg;
            }
        }
        Self(Rc::new(FusedChainFunctional::new_dimer(
            sigma1, sigma2, distance, version,
        )))
    }

    /// New functional for fused trimers.
    ///
    /// Parameters
    /// ----------
    /// sigma1: float
    ///     Diameter of the first segment.
    /// sigma2: float
    ///     Diameter of the second segment.
    /// sigma3: float
    ///     Diameter of the third segment.
    /// distance1: float
    ///     Distance between the centers of the first segments.
    /// distance2: float
    ///     Distance between the centers of the last segments.
    ///
    /// Returns
    /// -------
    /// FusedChainFunctional
    #[staticmethod]
    #[pyo3(text_signature = "(sigma1, sigma2, sigma3, distance1, distance2)")]
    fn new_trimer(
        sigma1: f64,
        sigma2: f64,
        sigma3: f64,
        distance1: f64,
        distance2: f64,
        kierlik_rosinberg: Option<bool>,
    ) -> Self {
        let mut version = FMTVersion::WhiteBear;
        if let Some(kierlik_rosinberg) = kierlik_rosinberg {
            if kierlik_rosinberg {
                version = FMTVersion::KierlikRosinberg;
            }
        }
        Self(Rc::new(FusedChainFunctional::new_trimer(
            sigma1, sigma2, sigma3, distance1, distance2, version,
        )))
    }

    /// New functional for fused homosegmented chains.
    ///
    /// Parameters
    /// ----------
    /// segments: int
    ///     NUmber of segments on the chain.
    /// sigma: float
    ///     Diameter of the first segment.
    /// distance: float
    ///     Distance between the centers of the first segments.
    ///
    /// Returns
    /// -------
    /// FusedChainFunctional
    #[staticmethod]
    #[pyo3(text_signature = "(segments, sigma, distance)")]
    fn new_homosegmented(
        segments: usize,
        sigma: f64,
        distance: f64,
        kierlik_rosinberg: Option<bool>,
    ) -> Self {
        let mut version = FMTVersion::WhiteBear;
        if let Some(kierlik_rosinberg) = kierlik_rosinberg {
            if kierlik_rosinberg {
                version = FMTVersion::KierlikRosinberg;
            }
        }
        Self(Rc::new(FusedChainFunctional::new_homosegmented(
            segments, sigma, distance, version,
        )))
    }
}

impl_equation_of_state!(PyFusedChainFunctional);
impl_state!(DFT<FusedChainFunctional>, PyFusedChainFunctional);

impl_pore!(FusedChainFunctional, PyFusedChainFunctional);

impl_adsorption!(FusedChainFunctional, PyFusedChainFunctional);

#[pymodule]
pub fn fused_chain(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(quantity))?;
    m.add_class::<PyFusedChainFunctional>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyPore1D>()?;
    m.add_class::<PyExternalPotential>()?;
    m.add_class::<PyAdsorption1D>()?;
    m.add_class::<PyDFTSolver>()?;
    m.add_class::<PyContributions>()?;

    py.run(
        "import sys; sys.modules['fused_chain.si'] = quantity",
        None,
        Some(m.dict()),
    )
}
