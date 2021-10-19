use core::marker::PhantomData;

use ark_ff::{Zero, One, Field, PrimeField, SquareRootField};
use ark_ff::vec::Vec;
use ark_poly::{Polynomial, UVPolynomial, univariate::{DensePolynomial}};
use crate::models::SWModelParameters;
use crate::ModelParameters;

use crate::hashing::map_to_curve_hasher::MapToCurve;
use crate::hashing::HashToCurveError;
use crate::{AffineCurve};
use crate::models::short_weierstrass_jacobian::GroupAffine;

use super::swu::{SWUParams, SWUMap};

/// Implementation for the WB hash to curve for the curves of Weierstrass form of y^2 = x^3 + a*x + b where b != 0 but a can be zero like BLS-381 curve. From [WB2019]
///
/// - [WB19] Wahby, R. S., & Boneh, D. (2019). Fast and simple constant-time
///   hashing to the bls12-381 elliptic curve. IACR Transactions on
///   Cryptographic Hardware and Embedded Systems, nil(nil), 154–179.
///   http://dx.doi.org/10.46586/tches.v2019.i4.154-179
///
///
pub trait WBParams<const ISO_DEG: usize> : SWModelParameters + Sized
    where [(); ISO_DEG + 1]: Sized, [(); ISO_DEG*2 + 1]: Sized,
{
    //The isogenouscurve should be defined over the same basefield but it can have different scalar field
    //type IsogenousCurveScalarField :
    type IsogenousCurve : SWUParams::<BaseField = <Self as ModelParameters>::BaseField>;
    //const ISO_DEG: usize;
    const PHI_X_NOM: [<Self::IsogenousCurve as ModelParameters>::BaseField; ISO_DEG + 1];
    const PHI_X_DEN: [<Self::IsogenousCurve as ModelParameters>::BaseField; ISO_DEG + 1];
    const PHI_Y_NOM: [<Self::IsogenousCurve as ModelParameters>::BaseField; ISO_DEG*2 + 1];
    const PHI_Y_DEN: [<Self::IsogenousCurve as ModelParameters>::BaseField; ISO_DEG*2 + 1];

    fn isogeny_map(domain_point: GroupAffine<Self::IsogenousCurve>) -> Result<GroupAffine<Self>, HashToCurveError>{

	let x_num : DensePolynomial<<Self::IsogenousCurve as ModelParameters>::BaseField> = <DensePolynomial<<Self::IsogenousCurve as ModelParameters>::BaseField>>::from_coefficients_slice(&Self::PHI_X_NOM[..]);
                                           
	let x_den : DensePolynomial<<Self::IsogenousCurve as ModelParameters>::BaseField> = <DensePolynomial<<Self::IsogenousCurve as ModelParameters>::BaseField>>::from_coefficients_slice(&Self::PHI_X_DEN[..]);


	let y_num : DensePolynomial<<Self::IsogenousCurve as ModelParameters>::BaseField> = <DensePolynomial<<Self::IsogenousCurve as ModelParameters>::BaseField>>::from_coefficients_slice(&Self::PHI_Y_NOM[..]);

	let y_den : DensePolynomial<<Self::IsogenousCurve as ModelParameters>::BaseField> = <DensePolynomial<<Self::IsogenousCurve as ModelParameters>::BaseField>>::from_coefficients_slice(&Self::PHI_Y_DEN[..]);

	
    let img_x = x_num.evaluate(&domain_point.x) / x_den.evaluate(&domain_point.x);
    let img_y = (y_num.evaluate(&domain_point.x) * domain_point.y) / y_den.evaluate(&domain_point.x);
    Ok(GroupAffine::<Self>::new(img_x, img_y, false))
    	

     }
	
}

pub struct WBMap<const ISO_DEG: usize, P: WBParams<ISO_DEG>>
where [(); ISO_DEG + 1]: Sized, [(); ISO_DEG*2 + 1]: Sized,
{
    pub domain: Vec<u8>,
    swu_field_curve_hasher: SWUMap<P::IsogenousCurve>,
    curve_params: PhantomData<fn() -> P>,

    //TODO: We should perhaps store the SWUMap object for efficiency
}

impl<const ISO_DEG: usize, P: WBParams<ISO_DEG>> MapToCurve<GroupAffine<P>> for WBMap<ISO_DEG, P>
    where [(); ISO_DEG + 1]: Sized, [(); ISO_DEG*2 + 1]: Sized,
{

    ///This is to verify if the provided WBparams makes sense, doesn't do much for now
    fn new_map_to_curve(domain: &[u8]) -> Result<Self, HashToCurveError>
    {
        Ok(
            WBMap {
                domain: domain.to_vec(),
                swu_field_curve_hasher : SWUMap::<P::IsogenousCurve>::new_map_to_curve(&domain).unwrap(),
                //map_to_isogenous_curve: SWUMap<Self::IsogenousCurve>;
                curve_params: PhantomData,
        })
    }
    
    /// Map random field point to a random curve point
    /// inspired from
    /// https://github.com/zcash/pasta_curves/blob/main/src/hashtocurve.rs
    fn map_to_curve(&self, element: <GroupAffine<P> as AffineCurve>::BaseField) -> Result<GroupAffine<P>, HashToCurveError>  {
        //fn hash_to_curve<'a>(domain_prefix: &'a str) -> Box<dyn Fn(&[u8]) -> Self + 'a> {

        //first we need to map the field point to the isogenous curve
        //let swu_field_curve_hasher = SWUMap::<P::IsogenousCurve>::new_map_to_curve(&[1]).unwrap();

        let point_on_isogenious_curve = self.swu_field_curve_hasher.map_to_curve(element).unwrap();
        P::isogeny_map(point_on_isogenious_curve)
    }

}



