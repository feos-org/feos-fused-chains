use crate::{FusedChainFunctional, FusedChainRecord};
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

/// Parameters for a single fused-chain molecule.
///
/// Parameters
/// ----------
/// sigma: numpy.ndarray[float]
///     Segment diameters.
/// bonds: [(int, int, float)]
///     List of bonds and corresponding bond lengths.
///
/// Returns
/// -------
/// FusedChainRecord
#[pyclass(name = "FusedChainRecord", unsendable)]
#[pyo3(text_signature = "(sigma, bonds)")]
#[derive(Clone)]
pub struct PyFusedChainRecord(FusedChainRecord);

#[pymethods]
impl PyFusedChainRecord {
    #[new]
    fn new(sigma: &PyArray1<f64>, bonds: Vec<(u32, u32, f64)>) -> Self {
        Self(FusedChainRecord::new(sigma.to_owned_array(), bonds))
    }
}

/// Helmholtz energy functional for fused chains.
#[pyclass(name = "FusedChainFunctional", unsendable)]
#[pyo3(text_signature = "(sigma, component_index, bonds, version=None)")]
#[derive(Clone)]
pub struct PyFusedChainFunctional(Rc<DFT<FusedChainFunctional>>);

#[pymethods]
impl PyFusedChainFunctional {
    /// Create a fused-chain Helmholtz energy functional from records.
    ///
    /// Parameters
    /// ----------
    /// records: [FusedChainRecords]
    ///     Pure component records.
    /// version: FMTVersion, optional
    ///     The specific version of FMT to be used.
    ///     Defaults to FMTVersion.WhiteBear
    ///
    /// Returns
    /// -------
    /// FusedChainFunctional
    #[staticmethod]
    #[pyo3(text_signature = "(records, version=None)")]
    fn from_records(records: Vec<PyFusedChainRecord>, version: Option<PyFMTVersion>) -> Self {
        Self(Rc::new(FusedChainFunctional::from_records(
            records.into_iter().map(|r| r.0).collect(),
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
    /// l12: float
    ///     Bond length between the two segments.
    /// version: FMTVersion, optional
    ///     The specific version of FMT to be used.
    ///     Defaults to FMTVersion.WhiteBear
    ///
    /// Returns
    /// -------
    /// FusedChainFunctional
    #[staticmethod]
    #[pyo3(text_signature = "(sigma1, sigma2, l12, version=None)")]
    fn new_dimer(sigma1: f64, sigma2: f64, l12: f64, version: Option<PyFMTVersion>) -> Self {
        Self(Rc::new(FusedChainFunctional::new_dimer(
            sigma1,
            sigma2,
            l12,
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
    /// l12: float
    ///     Bond length between the first segments.
    /// l23: float
    ///     Bond length between the last segments.
    /// version: FMTVersion, optional
    ///     The specific version of FMT to be used.
    ///     Defaults to FMTVersion.WhiteBear
    ///
    /// Returns
    /// -------
    /// FusedChainFunctional
    #[staticmethod]
    #[pyo3(text_signature = "(sigma1, sigma2, sigma3, l12, l23, version=None)")]
    fn new_trimer(
        sigma1: f64,
        sigma2: f64,
        sigma3: f64,
        l12: f64,
        l23: f64,
        version: Option<PyFMTVersion>,
    ) -> Self {
        Self(Rc::new(FusedChainFunctional::new_trimer(
            sigma1,
            sigma2,
            sigma3,
            l12,
            l23,
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
    ///     Diameter of the segments.
    /// l: float
    ///     Bond length of the chain.
    /// version: FMTVersion, optional
    ///     The specific version of FMT to be used.
    ///     Defaults to FMTVersion.WhiteBear
    ///
    /// Returns
    /// -------
    /// FusedChainFunctional
    #[staticmethod]
    #[pyo3(text_signature = "(segments, sigma, l, version=None)")]
    fn new_homosegmented(
        segments: usize,
        sigma: f64,
        l: f64,
        version: Option<PyFMTVersion>,
    ) -> Self {
        Self(Rc::new(FusedChainFunctional::new_homosegmented(
            segments,
            sigma,
            l,
            version.map(|v| v.0),
        )))
    }

    /// Calculate the packing fraction for the given partial densities.
    ///
    /// Parameters
    /// ----------
    /// partial_density: SIArray1
    ///     Partial densities of al components.
    ///
    /// Returns
    /// -------
    /// float
    fn packing_fraction(&self, partial_density: &PySIArray1) -> EosResult<f64> {
        self.0.functional.packing_fraction(partial_density)
    }
}

impl_equation_of_state!(PyFusedChainFunctional);
impl_state!(DFT<FusedChainFunctional>, PyFusedChainFunctional);

impl_pore!(FusedChainFunctional, PyFusedChainFunctional);

impl_adsorption!(FusedChainFunctional, PyFusedChainFunctional);

#[pymodule]
pub fn feos_fused_chains(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(quantity))?;
    m.add_class::<PyFusedChainFunctional>()?;
    m.add_class::<PyFusedChainRecord>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyPore1D>()?;
    m.add_class::<PyExternalPotential>()?;
    m.add_class::<PyAdsorption1D>()?;
    m.add_class::<PyDFTSolver>()?;
    m.add_class::<PyContributions>()?;
    m.add_class::<PyFMTVersion>()?;

    py.run(
        "import sys; sys.modules['feos_fused_chains.si'] = quantity",
        None,
        Some(m.dict()),
    )
}
