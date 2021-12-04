use crate::FusedChainFunctional;
use ::quantity::python::*;
use feos_core::python::{PyContributions, PyVerbosity};
use feos_core::*;
use feos_dft::adsorption::*;
use feos_dft::python::*;
use feos_dft::*;
use numpy::{PyArray1, PyArray2, PyArray4, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use quantity::si::*;
use std::rc::Rc;

/// Helmholtz energy functional for fused chains.
///
/// Parameters
/// ----------
/// sigma: numpy.ndarray[float]
///     Segment diameters.
/// component_index: numpy.ndarray[int]
///     Index of the component that each individual
///     segment is on.
/// bonds: [(int, int, float)]
///     List of bonds and corresponding bond lengths.
/// version: FMTVersion, optional
///     The specific version of FMT to be used.
///     Defaults to FMTVersion.WhiteBear
///
/// Returns
/// -------
/// FusedChainFunctional
#[pyclass(name = "FusedChainFunctional", unsendable)]
#[pyo3(text_signature = "(sigma, component_index, bonds, version=None)")]
#[derive(Clone)]
pub struct PyFusedChainFunctional(Rc<DFT<FusedChainFunctional>>);

#[pymethods]
impl PyFusedChainFunctional {
    #[new]
    fn new(
        sigma: &PyArray1<f64>,
        component_index: &PyArray1<usize>,
        bonds: Vec<(u32, u32, f64)>,
        version: Option<PyFMTVersion>,
    ) -> Self {
        Self(Rc::new(FusedChainFunctional::new(
            sigma.to_owned_array(),
            component_index.to_owned_array(),
            bonds,
            version.map(|v| v.0),
        )))
    }

    /// New functional for monomers.
    ///
    /// Parameters
    /// ----------
    /// sigma: float
    ///     Diameter of the monomer.
    /// version: FMTVersion, optional
    ///     The specific version of FMT to be used.
    ///     Defaults to FMTVersion.WhiteBear
    ///
    /// Returns
    /// -------
    /// FusedChainFunctional
    #[staticmethod]
    #[pyo3(text_signature = "(sigma, version=None)")]
    fn new_monomer(sigma: f64, version: Option<PyFMTVersion>) -> Self {
        Self(Rc::new(FusedChainFunctional::new_monomer(
            sigma,
            version.map(|v| v.0),
        )))
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
    /// version: FMTVersion, optional
    ///     The specific version of FMT to be used.
    ///     Defaults to FMTVersion.WhiteBear
    ///
    /// Returns
    /// -------
    /// FusedChainFunctional
    #[staticmethod]
    #[pyo3(text_signature = "(sigma1, sigma2, distance, version=None)")]
    fn new_dimer(sigma1: f64, sigma2: f64, distance: f64, version: Option<PyFMTVersion>) -> Self {
        Self(Rc::new(FusedChainFunctional::new_dimer(
            sigma1,
            sigma2,
            distance,
            version.map(|v| v.0),
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
    /// version: FMTVersion, optional
    ///     The specific version of FMT to be used.
    ///     Defaults to FMTVersion.WhiteBear
    ///
    /// Returns
    /// -------
    /// FusedChainFunctional
    #[staticmethod]
    #[pyo3(text_signature = "(sigma1, sigma2, sigma3, distance1, distance2, version=None)")]
    fn new_trimer(
        sigma1: f64,
        sigma2: f64,
        sigma3: f64,
        distance1: f64,
        distance2: f64,
        version: Option<PyFMTVersion>,
    ) -> Self {
        Self(Rc::new(FusedChainFunctional::new_trimer(
            sigma1,
            sigma2,
            sigma3,
            distance1,
            distance2,
            version.map(|v| v.0),
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
    /// version: FMTVersion, optional
    ///     The specific version of FMT to be used.
    ///     Defaults to FMTVersion.WhiteBear
    ///
    /// Returns
    /// -------
    /// FusedChainFunctional
    #[staticmethod]
    #[pyo3(text_signature = "(segments, sigma, distance, version=None)")]
    fn new_homosegmented(
        segments: usize,
        sigma: f64,
        distance: f64,
        version: Option<PyFMTVersion>,
    ) -> Self {
        Self(Rc::new(FusedChainFunctional::new_homosegmented(
            segments,
            sigma,
            distance,
            version.map(|v| v.0),
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
    m.add_class::<PyFMTVersion>()?;

    py.run(
        "import sys; sys.modules['fused_chain.si'] = quantity",
        None,
        Some(m.dict()),
    )
}
