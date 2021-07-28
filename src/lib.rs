use eos_core::EosResult;
use eos_dft::adsorption::FluidParameters;
use eos_dft::fundamental_measure_theory::FMTVersion;
use eos_dft::{
    FunctionalContribution, FunctionalContributionDual, HelmholtzEnergyFunctional, WeightFunction,
    WeightFunctionInfo, WeightFunctionShape, DFT,
};
use ndarray::{arr1, Array, Array1, ArrayView2, Axis, ScalarOperand, Slice, Zip};
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_6, PI};
use std::fmt;
use std::rc::Rc;

mod python;

const PI36M1: f64 = 1.0 / (36.0 * PI);
const N3_CUTOFF: f64 = 1e-5;

pub struct FusedChainParameters {
    n_segments: Array1<usize>,
    sigma: Array1<f64>,
    m: [Array1<f64>; 4],
    distance: Array1<f64>,
}

impl FluidParameters for FusedChainParameters {
    fn epsilon_k_ff(&self) -> &Array1<f64> {
        unreachable!()
    }

    fn sigma_ff(&self) -> &Array1<f64> {
        &self.sigma
    }

    fn m(&self) -> Array1<f64> {
        unreachable!()
    }
}

pub struct FusedChainFunctional {
    pub parameters: Rc<FusedChainParameters>,
    contributions: Vec<Box<dyn FunctionalContribution>>,
    max_eta: f64,
}

fn m_v(sigma1: f64, sigma2: f64, distance: f64) -> f64 {
    1.0 - 1.0 / (256.0 * (sigma1 * distance).powi(3))
        * (sigma2.powi(2) - sigma1.powi(2) - 4.0 * distance.powi(2) + 4.0 * distance * sigma1)
            .powi(2)
        * (sigma1.powi(2) - sigma2.powi(2) + 4.0 * distance.powi(2) + 8.0 * distance * sigma1)
}

fn m_a(sigma1: f64, sigma2: f64, distance: f64) -> f64 {
    (sigma1.powi(2) - sigma2.powi(2) + 4.0 * distance.powi(2) + 4.0 * distance * sigma1)
        / (8.0 * distance * sigma1)
}

impl FusedChainFunctional {
    pub fn new_dimer(sigma1: f64, sigma2: f64, distance: f64, version: FMTVersion) -> DFT<Self> {
        let n_segments = arr1(&[2]);
        let m_v = arr1(&[m_v(sigma1, sigma2, distance), m_v(sigma2, sigma1, distance)]);
        let m_a = arr1(&[m_a(sigma1, sigma2, distance), m_a(sigma2, sigma1, distance)]);
        let parameters = Rc::new(FusedChainParameters {
            n_segments: n_segments.clone(),
            sigma: arr1(&[sigma1, sigma2]),
            m: [Array1::ones(2), m_a.clone(), m_a.clone(), m_v],
            distance: arr1(&[distance]),
        });
        let mut contributions: Vec<Box<dyn FunctionalContribution>> = Vec::with_capacity(2);
        contributions.push(Box::new(FMTFunctional::new(&parameters, version)));
        contributions.push(Box::new(FusedSegmentChainFunctional::new(&parameters)));
        DFT::new(
            Self {
                parameters,
                contributions,
                max_eta: 0.5,
            },
            &Array::ones(m_a.len()),
            &n_segments,
        )
    }

    pub fn new_monomer(sigma: f64, version: FMTVersion) -> DFT<Self> {
        let n_segments = arr1(&[1]);
        let m_v = arr1(&[1.0]);
        let m_a = arr1(&[1.0]);
        let parameters = Rc::new(FusedChainParameters {
            n_segments: n_segments.clone(),
            sigma: arr1(&[sigma]),
            m: [Array1::ones(1), m_a.clone(), m_a.clone(), m_v],
            distance: arr1(&[]),
        });
        let mut contributions: Vec<Box<dyn FunctionalContribution>> = Vec::with_capacity(2);
        contributions.push(Box::new(FMTFunctional::new(&parameters, version)));
        contributions.push(Box::new(FusedSegmentChainFunctional::new(&parameters)));
        DFT::new(
            Self {
                parameters,
                contributions,
                max_eta: 0.5,
            },
            &Array::ones(m_a.len()),
            &n_segments,
        )
    }
}

impl HelmholtzEnergyFunctional for FusedChainFunctional {
    fn get_subset(&self, _component_list: &[usize]) -> DFT<Self> {
        unimplemented!()
    }

    fn compute_max_density(&self, moles: &Array1<f64>) -> f64 {
        let mut j = 0;
        let mut moles_segments = Array1::zeros(self.parameters.n_segments.sum());
        for i in 0..self.parameters.n_segments.len() {
            for _ in 0..self.parameters.n_segments[i] {
                moles_segments[j] = moles[i];
                j += 1
            }
        }
        self.max_eta * moles.sum()
            / (FRAC_PI_6
                * &self.parameters.m[3]
                * self.parameters.sigma.mapv(|s| s.powi(3))
                * moles_segments)
                .sum()
    }

    fn contributions(&self) -> &[Box<dyn FunctionalContribution>] {
        &self.contributions
    }

    fn isaft_weight_functions(&self, _: f64) -> Vec<WeightFunction<f64>> {
        let mut res =
            Vec::with_capacity(self.parameters.n_segments.sum() - self.parameters.n_segments.len());
        let mut j = 0;
        for ns in &self.parameters.n_segments {
            for _ in 0..ns - 1 {
                res.push(WeightFunction::new_scaled(
                    self.parameters.distance.clone(),
                    WeightFunctionShape::Delta,
                ));
                j += 1;
            }
            j += 1;
        }
        res
    }
}

struct FMTFunctional {
    parameters: Rc<FusedChainParameters>,
    version: FMTVersion,
}

impl FMTFunctional {
    fn new(parameters: &Rc<FusedChainParameters>, version: FMTVersion) -> Self {
        Self {
            parameters: parameters.clone(),
            version,
        }
    }
}

impl<N: DualNum<f64>> FunctionalContributionDual<N> for FMTFunctional {
    fn weight_functions(&self, _: N) -> WeightFunctionInfo<N> {
        let r = self.parameters.sigma.mapv(N::from) * 0.5;
        match self.version {
            FMTVersion::WhiteBear => {
                WeightFunctionInfo::new(self.parameters.n_segments.clone(), false)
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
                            prefactor: Zip::from(&self.parameters.m[1])
                                .and(&r)
                                .map_collect(|&m, &r| r.recip() * m / (4.0 * PI)),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::Delta,
                        },
                        true,
                    )
                    .add(
                        WeightFunction {
                            prefactor: self.parameters.m[2].mapv(N::from),
                            kernel_radius: r.clone(), // * &self.parameters.m_a.mapv(f64::sqrt),
                            shape: WeightFunctionShape::Delta,
                        },
                        true,
                    )
                    .add(
                        WeightFunction {
                            prefactor: self.parameters.m[3].mapv(N::from),
                            kernel_radius: r.clone(), // * &self.parameters.m_v.mapv(f64::cbrt),
                            shape: WeightFunctionShape::Theta,
                        },
                        true,
                    )
                    .add(
                        WeightFunction {
                            prefactor: Zip::from(&self.parameters.m[3])
                                .and(&r)
                                .map_collect(|&m, &r| r.recip() * m / (4.0 * PI)),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::DeltaVec,
                        },
                        true,
                    )
                    .add(
                        WeightFunction {
                            prefactor: self.parameters.m[3].mapv(N::from),
                            kernel_radius: r.clone(),
                            shape: WeightFunctionShape::DeltaVec,
                        },
                        true,
                    )
            }
            FMTVersion::KierlikRosinberg => {
                WeightFunctionInfo::new(self.parameters.n_segments.clone(), false).extend(
                    vec![
                        WeightFunctionShape::KR0,
                        WeightFunctionShape::KR1,
                        WeightFunctionShape::Delta,
                        WeightFunctionShape::Theta,
                    ]
                    .into_iter()
                    .zip(self.parameters.m.iter())
                    .map(|(s, m)| WeightFunction {
                        prefactor: m.mapv(N::from),
                        kernel_radius: r.clone(),
                        shape: s,
                    })
                    .collect(),
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
        let n1v = weighted_densities.slice_axis(Axis(0), Slice::new(4, Some(4 + dim), 1));
        let n2v = weighted_densities.slice_axis(Axis(0), Slice::new(4 + dim, Some(4 + 2 * dim), 1));

        // auxiliary variables
        let ln31 = n3.mapv(|n3| (-n3).ln_1p());
        let n3rec = n3.mapv(|n3| n3.recip());
        let n3m1 = n3.mapv(|n3| -n3 + 1.0);
        let n3m1rec = n3m1.mapv(|n3m1| n3m1.recip());

        // White-Bear FMT
        // use Taylor expansion for f3 at low densities to avoid numerical issues
        let mut f3 = (&n3m1 * &n3m1 * &ln31 + &n3) * &n3rec * n3rec * &n3m1rec * &n3m1rec;
        f3.iter_mut().zip(n3).for_each(|(f3, &n3)| {
            if n3.re() < N3_CUTOFF {
                *f3 = (((n3 * 35.0 / 6.0 + 4.8) * n3 + 3.75) * n3 + 8.0 / 3.0) * n3 + 1.5;
            }
        });
        Ok(-(&n0 * &ln31)
            + (&n1 * &n2 - (&n1v * &n2v).sum_axis(Axis(0))) * &n3m1rec
            + (&n2 * &n2 - (&n2v * &n2v).sum_axis(Axis(0)) * 3.0) * &n2 * PI36M1 * f3)
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
        WeightFunctionInfo::new(self.parameters.n_segments.clone(), true)
            .add(
                WeightFunction {
                    prefactor: self.parameters.m[2].mapv(|m| m.into()) / (&d * 8.0),
                    kernel_radius: d.clone(),
                    shape: WeightFunctionShape::Theta,
                },
                true,
            )
            // .add(
            //     WeightFunction {
            //         prefactor: self.parameters.m_a.mapv(|m| (m / 24.0).into()),
            //         kernel_radius: d.clone(),
            //         shape: WeightFunctionShape::Delta,
            //     },
            //     true,
            // )
            .add(
                WeightFunction {
                    prefactor: self.parameters.m[3].mapv(|m| (m / 8.0).into()),
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
        let p = &self.parameters;
        // number of segments
        let n = weighted_densities.shape()[0] - 2;

        // weighted densities
        let rho = weighted_densities.slice_axis(Axis(0), Slice::new(0, Some(n as isize), 1));
        let zeta2 = weighted_densities.index_axis(Axis(0), n);
        let zeta3 = weighted_densities.index_axis(Axis(0), n + 1);

        let z3i = zeta3.mapv(|z3| (-z3 + 1.0).recip());
        let mut j = 0;
        let mut phi = Array::zeros(zeta2.raw_dim());
        for ns in self.parameters.n_segments.iter() {
            for _ in 0..ns - 1 {
                // cavity correlation
                let mij = 0.5 * (p.m[1][j] + p.m[1][j + 1]);
                let dij = p.sigma[j] * p.sigma[j + 1] / (p.sigma[j] + p.sigma[j + 1]);
                let z2d = zeta2.mapv(|z2| z2 * dij);
                let yi = &z2d * &z3i * &z3i * (z2d * &z3i * 2.0 + 3.0) + &z3i;

                // Helmholtz energy density
                let rhom = (&rho.index_axis(Axis(0), j) + &rho.index_axis(Axis(0), j + 1)) * 0.5;
                phi = phi - yi.map(N::ln) * rhom * mij;
                j += 1;
            }
            j += 1
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
    use eos_core::StateBuilder;
    use eos_dft::adsorption::{AdsorptionProfile, ExternalPotential};
    use quantity::si::{ANGSTROM, KELVIN, METER, MOL};

    #[test]
    fn test_fused_chain_functional() -> EosResult<()> {
        let func = Rc::new(FusedChainFunctional::new_dimer(
            1.0,
            1.0,
            1.0,
            FMTVersion::WhiteBear,
        ));
        let bulk = StateBuilder::new(&func)
            .temperature(100.0 * KELVIN)
            .density(0.05 * MOL / METER.powi(3))
            .build()?;
        AdsorptionProfile::new_slit_pore(
            &bulk,
            1024,
            100.0 * ANGSTROM,
            &ExternalPotential::HardWall { sigma_ss: 1.0 },
            &func.functional.parameters,
            None,
        )?
        .solve(None)?;
        Ok(())
    }
}
