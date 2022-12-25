use ark_std::{vec, vec::Vec};

macro_rules! adc {
    ($a:expr, $b:expr, &mut $carry:expr$(,)?) => {{
        let tmp = ($a as u128) + ($b as u128) + ($carry as u128);
        $carry = (tmp >> 64) as u64;
        tmp as u64
    }};
}

/// Sets a = a + b + carry, and returns the new carry.
#[inline(always)]
#[allow(unused_mut)]
#[doc(hidden)]
pub fn adc(a: &mut u64, b: u64, carry: u64) -> u64 {
    let tmp = *a as u128 + b as u128 + carry as u128;
    *a = tmp as u64;
    (tmp >> 64) as u64
}

/// Sets a = a + b + carry, and returns the new carry.
#[inline(always)]
#[allow(unused_mut)]
#[doc(hidden)]
pub fn adc_for_add_with_carry(a: &mut u64, b: u64, carry: u8) -> u8 {
    #[cfg(all(target_arch = "x86_64", feature = "asm"))]
    #[allow(unsafe_code)]
    unsafe {
        use core::arch::x86_64::_addcarry_u64;
        _addcarry_u64(carry, *a, b, a)
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "asm")))]
    {
        let tmp = *a as u128 + b as u128 + carry as u128;
        *a = tmp as u64;
        (tmp >> 64) as u8
    }
}

/// Calculate a + b + carry, returning the sum
#[inline(always)]
#[doc(hidden)]
pub fn adc_no_carry(a: u64, b: u64, carry: &mut u64) -> u64 {
    let tmp = a as u128 + b as u128 + *carry as u128;
    tmp as u64
}

#[macro_export]
macro_rules! sbb {
    ($a:expr, $b:expr, &mut $borrow:expr$(,)?) => {{
        let tmp = (1u128 << 64) + ($a as u128) - ($b as u128) - ($borrow as u128);
        $borrow = if tmp >> 64 == 0 { 1 } else { 0 };
        tmp as u64
    }};
}

/// Sets a = a - b - borrow, and returns the borrow.
#[inline(always)]
#[allow(unused_mut)]
pub(crate) fn sbb(a: &mut u64, b: u64, borrow: u64) -> u64 {
    let tmp = (1u128 << 64) + (*a as u128) - (b as u128) - (borrow as u128);
    *a = tmp as u64;
    if tmp >> 64 == 0 {
        1
    } else {
        0
    }
}

/// Sets a = a - b - borrow, and returns the borrow.
#[inline(always)]
#[allow(unused_mut)]
#[doc(hidden)]
pub fn sbb_for_sub_with_borrow(a: &mut u64, b: u64, borrow: u8) -> u8 {
    #[cfg(all(target_arch = "x86_64", feature = "asm"))]
    #[allow(unsafe_code)]
    unsafe {
        use core::arch::x86_64::_subborrow_u64;
        _subborrow_u64(borrow, *a, b, a)
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "asm")))]
    {
        let tmp = (1u128 << 64) + (*a as u128) - (b as u128) - (borrow as u128);
        *a = tmp as u64;
        u8::from(tmp >> 64 == 0)
    }
}

/// Calculate a + b * c, returning the lower 64 bits of the result and setting
/// `carry` to the upper 64 bits.
#[inline(always)]
#[doc(hidden)]
pub fn mac(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    let tmp = (a as u128) + (b as u128 * c as u128);
    *carry = (tmp >> 64) as u64;
    tmp as u64
}

/// Calculate a + b * c, discarding the lower 64 bits of the result and setting
/// `carry` to the upper 64 bits.
#[inline(always)]
#[doc(hidden)]
pub fn mac_discard(a: u64, b: u64, c: u64, carry: &mut u64) {
    let tmp = (a as u128) + (b as u128 * c as u128);
    *carry = (tmp >> 64) as u64;
}

macro_rules! mac_with_carry {
    ($a:expr, $b:expr, $c:expr, &mut $carry:expr$(,)?) => {{
        let tmp = ($a as u128) + ($b as u128 * $c as u128) + ($carry as u128);
        $carry = (tmp >> 64) as u64;
        tmp as u64
    }};
}

macro_rules! mac {
    ($a:expr, $b:expr, $c:expr, &mut $carry:expr$(,)?) => {{
        let tmp = ($a as u128) + ($b as u128 * $c as u128);
        $carry = (tmp >> 64) as u64;
        tmp as u64
    }};
}

/// Calculate a + (b * c) + carry, returning the least significant digit
/// and setting carry to the most significant digit.
#[inline(always)]
#[doc(hidden)]
pub fn mac_with_carry(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    let tmp = (a as u128) + (b as u128 * c as u128) + (*carry as u128);
    *carry = (tmp >> 64) as u64;
    tmp as u64
}

/// Calculate a + 2*(b * c) + carry_lo  (carry_hi << 64), returning the least significant digit
/// and setting carry0 to the 2nd-most significant digit, and carry1 to most significant digit
/// TODO carry1 could be a bool/u8?
#[inline(always)]
#[doc(hidden)]
pub fn mul_double_add_with_carry_2(
    a: u64,
    b: u64,
    c: u64,
    carry_lo: &mut u64,
    carry_hi: &mut u64,
) -> u64 {
    let mut tmp1 = (b as u128) * (c as u128);
    // it will spill a bit after *2 if the topmost bit is 1
    let doubling_high_carry = (tmp1 >> 127) as u64;
    // multiply by 2
    tmp1 <<= 1;
    // get the topmost bits of tmp1
    let doubling_low_carry = tmp1 >> 64;
    let doubling_bottom = tmp1 as u64;

    // add the rest. Every element here should fit into a single u64 digit.
    let tmp3 = (a as u128) + (doubling_bottom as u128) + (*carry_lo as u128);

    let tmp4 = (tmp3 >> 64) + (doubling_low_carry) + (*carry_hi as u128);
    // at this point, either tmp4 or doubling_high_carry are 1, but not both
    // assert!((tmp4 >> 64) + (doubling_high_carry as u128) <= 1);
    *carry_hi = (((tmp4 >> 64) as u64) + doubling_high_carry) as u64;

    *carry_lo = tmp4 as u64;

    tmp3 as u64
}

/// Compute the NAF (non-adjacent form) of num
pub fn find_naf(num: &[u64]) -> Vec<i8> {
    let is_zero = |num: &[u64]| num.iter().all(|x| *x == 0u64);
    let is_odd = |num: &[u64]| num[0] & 1 == 1;
    let sub_noborrow = |num: &mut [u64], z: u64| {
        let mut other = vec![0u64; num.len()];
        other[0] = z;
        let mut borrow = 0;

        for (a, b) in num.iter_mut().zip(other) {
            borrow = sbb(a, b, borrow);
        }
    };
    let add_nocarry = |num: &mut [u64], z: u64| {
        let mut other = vec![0u64; num.len()];
        other[0] = z;
        let mut carry = 0;

        for (a, b) in num.iter_mut().zip(other) {
            carry = adc(a, b, carry);
        }
    };
    let div2 = |num: &mut [u64]| {
        let mut t = 0;
        for i in num.iter_mut().rev() {
            let t2 = *i << 63;
            *i >>= 1;
            *i |= t;
            t = t2;
        }
    };

    let mut num = num.to_vec();
    let mut res = vec![];

    while !is_zero(&num) {
        let z: i8;
        if is_odd(&num) {
            z = 2 - (num[0] % 4) as i8;
            if z >= 0 {
                sub_noborrow(&mut num, z as u64)
            } else {
                add_nocarry(&mut num, (-z) as u64)
            }
        } else {
            z = 0;
        }
        res.push(z);
        div2(&mut num);
    }

    res
}

/// We define relaxed NAF as a variant of NAF with a very small tweak.
///
/// Note that the cost of scalar multiplication grows with the length of the sequence (for doubling)
/// plus the Hamming weight of the sequence (for addition, or subtraction).
///
/// NAF is optimizing for the Hamming weight only and therefore can be suboptimal.
/// For example, NAF may generate a sequence (in little-endian) of the form ...0 -1 0 1.
///
/// This can be rewritten as ...0 1 1 to avoid one doubling, at the cost that we are making an
/// exception of non-adjacence for the most significant bit.
///
/// Since this representation is no longer a strict NAF, we call it "relaxed NAF".
pub fn find_relaxed_naf(num: &[u64]) -> Vec<i8> {
    let mut res = find_naf(num);

    let len = res.len();
    if res[len - 2] == 0 && res[len - 3] == -1 {
        res[len - 3] = 1;
        res[len - 2] = 1;
        res.resize(len - 1, 0);
    }

    res
}

#[cfg(test)]
mod tests {
    use ark_std::{One, UniformRand, Zero};
    use num_bigint::{BigInt, BigUint};

    use super::*;

    #[test]
    fn test_find_relaxed_naf_usefulness() {
        let vec = find_naf(&[12u64]);
        assert_eq!(vec.len(), 5);

        let vec = find_relaxed_naf(&[12u64]);
        assert_eq!(vec.len(), 4);
    }

    #[test]
    fn test_find_relaxed_naf_correctness() {
        let mut rng = ark_std::test_rng();

        for _ in 0..10 {
            let num = [
                u64::rand(&mut rng),
                u64::rand(&mut rng),
                u64::rand(&mut rng),
                u64::rand(&mut rng),
            ];
            let relaxed_naf = find_relaxed_naf(&num);

            let test = {
                let mut sum = BigInt::zero();
                let mut cur = BigInt::one();
                for v in relaxed_naf {
                    sum += cur.clone() * v;
                    cur *= 2;
                }
                sum
            };

            let test_expected = {
                let mut sum = BigInt::zero();
                let mut cur = BigInt::one();
                for v in num.iter() {
                    sum += cur.clone() * v;
                    cur <<= 64;
                }
                sum
            };

            assert_eq!(test, test_expected);
        }
    }

    #[test]
    fn test_mul_double_add_with_carry_2() {
        // random tests, carry1 = 0
        let mut rng = ark_std::test_rng();
        for _ in 0..100 {
            let mut carry1 = 0u64;
            let mut carry0 = u64::rand(&mut rng);
            let a = u64::rand(&mut rng);
            let b = u64::rand(&mut rng);
            let c = u64::rand(&mut rng);

            let mut expected: BigUint = BigUint::zero();
            expected += BigUint::from(a);
            expected += BigUint::from(2u64) * BigUint::from(b) * BigUint::from(c);
            expected += BigUint::from(carry0);
            expected += BigUint::from(carry1) << 64;

            let res = mul_double_add_with_carry_2(a, b, c, &mut carry0, &mut carry1);
            let actual =
                BigUint::from(res) + (BigUint::from(carry0) << 64) + (BigUint::from(carry1) << 128);
            assert_eq!(actual, expected);
        }

        // random tests, carry1 = 1
        for _ in 0..100 {
            let mut carry1 = 1u64;
            let mut carry0 = u64::rand(&mut rng);
            let a = u64::rand(&mut rng);
            let b = u64::rand(&mut rng);
            let c = u64::rand(&mut rng);

            let mut expected: BigUint = BigUint::zero();
            expected += BigUint::from(a);
            expected += BigUint::from(2u64) * BigUint::from(b) * BigUint::from(c);
            expected += BigUint::from(carry0);
            expected += BigUint::from(carry1) << 64;

            let res = mul_double_add_with_carry_2(a, b, c, &mut carry0, &mut carry1);
            let actual =
                BigUint::from(res) + (BigUint::from(carry0) << 64) + (BigUint::from(carry1) << 128);
            assert_eq!(actual, expected);
        }
    }
}
