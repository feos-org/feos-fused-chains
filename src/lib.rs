#![warn(clippy::all)]
use feos_core::EosResult;
use feos_dft::adsorption::FluidParameters;
use feos_dft::fundamental_measure_theory::FMTVersion;
use feos_dft::{
    FunctionalContribution, FunctionalContributionDual, HelmholtzEnergyFunctional, WeightFunction,
    WeightFunctionInfo, WeightFunctionShape, DFT,
};
use ndarray::{arr1, Array, Array1, ArrayView2, Axis, ScalarOperand, Slice, Zip};
use num_dual::DualNum;
use petgraph::graph::{Graph, UnGraph};
use petgraph::visit::EdgeRef;
use std::f64::consts::{FRAC_PI_6, PI};
use std::fmt;
use std::rc::Rc;

#[cfg(feature = "python")]
mod python;

const PI36M1: f64 = 1.0 / (36.0 * PI);
const N3_CUTOFF: f64 = 1e-5;

pub struct FusedChainParameters {
    sigma: Array1<f64>,
    a: Array1<f64>,
    v: Array1<f64>,
    l: UnGraph<(), f64>,
    component_index: Array1<usize>,
}

pub struct FusedChainFunctional {
    pub parameters: Rc<FusedChainParameters>,
    contributions: Vec<Box<dyn FunctionalContribution>>,
    max_eta: f64,
}

impl FusedChainFunctional {
    fn new(
        sigma: Array1<f64>,
        component_index: Array1<usize>,
        bonds: Vec<(u32, u32, f64)>,
        version: Option<FMTVersion>,
    ) -> DFT<Self> {
        let segments = sigma.len();

        let mut l = Graph::default();
        l.extend_with_edges(bonds.into_iter());

        let mut a = Array::ones(segments);
        let mut v = Array::ones(segments);
        for n in l.node_indices() {
            let sigma1 = sigma[n.index()];
            for e in l.edges(n) {
                let sigma2 = sigma[e.target().index()];
                let l12 = e.weight();
                let delta12 = (sigma1.powi(2) - sigma2.powi(2) + 4.0 * l12.powi(2)) / (8.0 * l12);
                a[n.index()] -= 0.5 * (1.0 - 2.0 * delta12 / sigma1);
                v[n.index()] -=
                    0.5 * (1.0 - 3.0 * delta12 / sigma1 + 4.0 * (delta12 / sigma1).powi(3));
            }
        }

        let parameters = Rc::new(FusedChainParameters {
            sigma,
            a,
            v,
            l,
            component_index: component_index.clone(),
        });
        let mut contributions: Vec<Box<dyn FunctionalContribution>> = Vec::with_capacity(2);
        contributions.push(Box::new(FMTFunctional::new(&parameters, version)));
        if segments > 1 {
            contributions.push(Box::new(FusedSegmentChainFunctional::new(&parameters)));
        }
        DFT::new_heterosegmented(
            Self {
                parameters,
                contributions,
                max_eta: 0.5,
            },
            &component_index,
        )
    }

    pub fn new_monomer(sigma: f64, version: Option<FMTVersion>) -> DFT<Self> {
        Self::new(arr1(&[sigma]), arr1(&[0]), vec![], version)
    }

    pub fn new_dimer(sigma1: f64, sigma2: f64, l12: f64, version: Option<FMTVersion>) -> DFT<Self> {
        Self::new(
            arr1(&[sigma1, sigma2]),
            arr1(&[0, 0]),
            vec![(0, 1, l12)],
            version,
        )
    }

    pub fn new_trimer(
        sigma1: f64,
        sigma2: f64,
        sigma3: f64,
        l12: f64,
        l23: f64,
        version: Option<FMTVersion>,
    ) -> DFT<Self> {
        Self::new(
            arr1(&[sigma1, sigma2, sigma3]),
            arr1(&[0, 0, 0]),
            vec![(0, 1, l12), (1, 2, l23)],
            version,
        )
    }

    pub fn new_homosegmented(
        segments: usize,
        sigma: f64,
        l: f64,
        version: Option<FMTVersion>,
    ) -> DFT<Self> {
        Self::new(
            Array1::from_elem(segments, sigma),
            Array1::from_elem(segments, 0),
            (0..segments as u32 - 1)
                .into_iter()
                .map(|i| (i, i + 1, l))
                .collect(),
            version,
        )
    }
}

impl HelmholtzEnergyFunctional for FusedChainFunctional {
    fn subset(&self, _component_list: &[usize]) -> DFT<Self> {
        unimplemented!()
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let moles_segments = self.parameters.component_index.mapv(|c| moles[c]);
        self.max_eta * moles.sum()
            / (FRAC_PI_6
                * &self.parameters.v
                * self.parameters.sigma.mapv(|s| s.powi(3))
                * moles_segments)
                .sum()
    }

    fn contributions(&self) -> &[Box<dyn FunctionalContribution>] {
        &self.contributions
    }

    fn bond_lengths(&self, _: f64) -> UnGraph<(), f64> {
        self.parameters.l.clone()
    }
}

impl FluidParameters for FusedChainFunctional {
    fn epsilon_k_ff(&self) -> Array1<f64> {
        Array::zeros(self.parameters.sigma.len())
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        &self.parameters.sigma
    }

    fn m(&self) -> Array1<f64> {
        Array1::ones(self.parameters.sigma.len())
    }
}

struct FMTFunctional {
    parameters: Rc<FusedChainParameters>,
    version: FMTVersion,
}

impl FMTFunctional {
    fn new(parameters: &Rc<FusedChainParameters>, version: Option<FMTVersion>) -> Self {
        Self {
            parameters: parameters.clone(),
            version: version.unwrap_or(FMTVersion::WhiteBear),
        }
    }
}

impl<N: DualNum<f64>> FunctionalContributionDual<N> for FMTFunctional {
    fn weight_functions(&self, _: N) -> WeightFunctionInfo<N> {
        let r = self.parameters.sigma.mapv(N::from) * 0.5;
        match self.version {
            FMTVersion::WhiteBear | FMTVersion::AntiSymWhiteBear => {
                WeightFunctionInfo::new(self.parameters.component_index.clone(), false)
                    .add(
                        WeightFunction {
                            prefactor: r.mapv(|r| r.powi(-2) / (4.0 * PI)),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::Delta,
                        },
                        true,
                    )
                    .add(
                        WeightFunction {
                            prefactor: Zip::from(&self.parameters.a)
                                .and(&r)
                                .map_collect(|&a, &r| r.recip() * a / (4.0 * PI)),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::Delta,
                        },
                        true,
                    )
                    .add(
                        WeightFunction {
                            prefactor: self.parameters.a.mapv(N::from),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::Delta,
                        },
                        true,
                    )
                    .add(
                        WeightFunction {
                            prefactor: self.parameters.v.mapv(N::from),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::Theta,
                        },
                        true,
                    )
                    .add(
                        WeightFunction {
                            prefactor: Zip::from(&self.parameters.v)
                                .and(&r)
                                .map_collect(|&m, &r| r.recip() * m / (4.0 * PI)),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::DeltaVec,
                        },
                        true,
                    )
                    .add(
                        WeightFunction {
                            prefactor: self.parameters.v.mapv(N::from),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::DeltaVec,
                        },
                        true,
                    )
            }
            FMTVersion::KierlikRosinberg => {
                WeightFunctionInfo::new(self.parameters.component_index.clone(), false)
                    .add(
                        WeightFunction {
                            prefactor: Array::ones(r.len()),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::KR0,
                        },
                        true,
                    )
                    .add(
                        WeightFunction {
                            prefactor: self.parameters.a.mapv(N::from),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::KR1,
                        },
                        true,
                    )
                    .add(
                        WeightFunction {
                            prefactor: self.parameters.a.mapv(N::from),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::Delta,
                        },
                        true,
                    )
                    .add(
                        WeightFunction {
                            prefactor: self.parameters.v.mapv(N::from),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::Theta,
                        },
                        true,
                    )
            }
        }
    }

    fn calculate_helmholtz_energy_density(
        &self,
        _: N,
        weighted_densities: ArrayView2<N>,
    ) -> EosResult<Array1<N>> {
        // number of dimensions
        let dim = ((weighted_densities.shape()[0] - 4) / 2) as isize;

        // weighted densities
        let n0 = weighted_densities.index_axis(Axis(0), 0);
        let n1 = weighted_densities.index_axis(Axis(0), 1);
        let n2 = weighted_densities.index_axis(Axis(0), 2);
        let n3 = weighted_densities.index_axis(Axis(0), 3);

        let (n1n2, n2n2) = match self.version {
            FMTVersion::WhiteBear => {
                let n1v = weighted_densities.slice_axis(Axis(0), Slice::new(4, Some(4 + dim), 1));
                let n2v = weighted_densities
                    .slice_axis(Axis(0), Slice::new(4 + dim, Some(4 + 2 * dim), 1));
                (
                    &n1 * &n2 - (&n1v * &n2v).sum_axis(Axis(0)),
                    &n2 * &n2 - (&n2v * &n2v).sum_axis(Axis(0)) * 3.0,
                )
            }
            FMTVersion::KierlikRosinberg => (&n1 * &n2, &n2 * &n2),
            FMTVersion::AntiSymWhiteBear => unimplemented!(),
        };

        // auxiliary variables
        let ln31 = n3.mapv(|n3| (-n3).ln_1p());
        let n3rec = n3.mapv(|n3| n3.recip());
        let n3m1 = n3.mapv(|n3| -n3 + 1.0);
        let n3m1rec = n3m1.mapv(|n3m1| n3m1.recip());

        // use Taylor expansion for f3 at low densities to avoid numerical issues
        let mut f3 = (&n3m1 * &n3m1 * &ln31 + n3) * &n3rec * n3rec * &n3m1rec * &n3m1rec;
        f3.iter_mut().zip(n3).for_each(|(f3, &n3)| {
            if n3.re() < N3_CUTOFF {
                *f3 = (((n3 * 35.0 / 6.0 + 4.8) * n3 + 3.75) * n3 + 8.0 / 3.0) * n3 + 1.5;
            }
        });
        Ok(-(&n0 * &ln31) + n1n2 * &n3m1rec + n2n2 * n2 * PI36M1 * f3)
    }
}

impl fmt::Display for FMTFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FMT functional (WB)")
    }
}

struct FusedSegmentChainFunctional {
    parameters: Rc<FusedChainParameters>,
}

impl FusedSegmentChainFunctional {
    fn new(parameters: &Rc<FusedChainParameters>) -> Self {
        Self {
            parameters: parameters.clone(),
        }
    }
}

impl<N: DualNum<f64> + ScalarOperand> FunctionalContributionDual<N>
    for FusedSegmentChainFunctional
{
    fn weight_functions(&self, _: N) -> WeightFunctionInfo<N> {
        let d = self.parameters.sigma.mapv(N::from);
        WeightFunctionInfo::new(self.parameters.component_index.clone(), true)
            .add(
                WeightFunction {
                    prefactor: self.parameters.a.mapv(N::from) / (&d * 8.0),
                    kernel_radius: d.clone(),
                    shape: WeightFunctionShape::Theta,
                },
                true,
            )
            .add(
                WeightFunction {
                    prefactor: self.parameters.v.mapv(|m| (m / 8.0).into()),
                    kernel_radius: d,
                    shape: WeightFunctionShape::Theta,
                },
                true,
            )
    }

    fn calculate_helmholtz_energy_density(
        &self,
        _: N,
        weighted_densities: ArrayView2<N>,
    ) -> EosResult<Array1<N>> {
        // number of segments
        let n = weighted_densities.shape()[0] - 2;

        // weighted densities
        let rho = weighted_densities.slice_axis(Axis(0), Slice::new(0, Some(n as isize), 1));
        let zeta2 = weighted_densities.index_axis(Axis(0), n);
        let zeta3 = weighted_densities.index_axis(Axis(0), n + 1);

        let z3i = zeta3.mapv(|z3| (-z3 + 1.0).recip());

        let mut phi = Array::zeros(zeta2.raw_dim());
        for (rho_i, i) in rho.axis_iter(Axis(0)).zip(self.parameters.l.node_indices()) {
            let edges = self.parameters.l.edges(i);
            let y = edges
                .map(|e| {
                    let l = e.weight();
                    let s1 = self.parameters.sigma[e.source().index()];
                    let s2 = self.parameters.sigma[e.target().index()];
                    let delta = (4.0 * l.powi(2) - (s1 - s2).powi(2)) / (4.0 * l);
                    let z2l = zeta2.mapv(|z2| z2 * delta);
                    &z2l * &z3i * &z3i * (z2l * &z3i * 0.5 + 1.5) + &z3i
                })
                .reduce(|acc, y| acc * y);
            if let Some(y) = y {
                phi -= &(y.map(N::ln) * rho_i * 0.5);
            }
        }

        Ok(phi)
    }
}

impl fmt::Display for FusedSegmentChainFunctional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hard chain segment functional")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use feos_core::StateBuilder;
    use feos_dft::adsorption::{ExternalPotential, Pore1D, PoreSpecification};
    use feos_dft::AxisGeometry;
    use quantity::si::{ANGSTROM, KELVIN, KILO, METER, MOL};

    #[test]
    fn test_fused_chain_functional() -> EosResult<()> {
        let func = Rc::new(FusedChainFunctional::new_trimer(
            1.0, 1.0, 1.0, 1.0, 1.0, None,
        ));
        let bulk = StateBuilder::new(&func)
            .temperature(100.0 * KELVIN)
            .density(272.3 * KILO * MOL / METER.powi(3))
            .build()?;
        Pore1D::new(
            &func,
            AxisGeometry::Cartesian,
            100.0 * ANGSTROM,
            ExternalPotential::HardWall { sigma_ss: 1.0 },
            Some(1024),
            None,
        )
        .initialize(&bulk, None)?
        .solve(None)?;
        Ok(())
    }
}
