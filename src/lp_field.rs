// use num_bigint::BigUint;
// use num_traits::ToPrimitive;
// use serde::{Serialize, Deserialize};
// use std::ops::{Add, Mul};

// pub const MODULUS_64: u64 = 9223372036854775783u64;

// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// pub struct FieldElm {
//     pub value: u64,
// }

// impl FieldElm {
//     /// Create a new field element from a given value (mod MODULUS_64)
//     pub fn from(val: u64) -> Self {
//         Self { value: val % MODULUS_64 }
//     }

//     /// Addition in the field
//     pub fn add(&self, other: &FieldElm) -> FieldElm {
//         FieldElm::from(self.value + other.value)
//     }

//     /// Negation in the field
//     pub fn neg(&self) -> FieldElm {
//         if self.value == 0 {
//             FieldElm::from(0)
//         } else {
//             FieldElm::from(MODULUS_64 - self.value)
//         }
//     }

//     /// Returns 1 in the field
//     pub fn one() -> Self {
//         Self::from(1u64)
//     }

//     /// Returns 0 in the field
//     pub fn zero() -> Self {
//         Self::from(0u64)
//     }

//     /// Computes modular inverse using Fermat's little theorem: a^(p-2) mod p
//     pub fn mod_inverse(&self) -> FieldElm {
//         let base = BigUint::from(self.value);
//         let exponent = BigUint::from(MODULUS_64 - 2);
//         let modulus_big = BigUint::from(MODULUS_64);

//         let result = base.modpow(&exponent, &modulus_big);
//         let value = result.to_u64_digits().first().copied().unwrap_or(0);
        
//         FieldElm::from(value)
//     }
// }

// //
// // Safe and unified multiplication implementations
// //

// impl<'a, 'b> Mul<&'b FieldElm> for &'a FieldElm {
//     type Output = FieldElm;

//     fn mul(self, rhs: &'b FieldElm) -> FieldElm {
//         let big_self = BigUint::from(self.value);
//         let big_rhs = BigUint::from(rhs.value);
//         let result = (big_self * big_rhs) % BigUint::from(MODULUS_64);
//         let value = result.to_u64_digits().first().copied().unwrap_or(0);
//         FieldElm::from(value)
//     }
// }

// impl<'a> Mul<FieldElm> for &'a FieldElm {
//     type Output = FieldElm;

//     fn mul(self, rhs: FieldElm) -> FieldElm {
//         self * &rhs
//     }
// }

// impl<'a> Mul<&'a FieldElm> for FieldElm {
//     type Output = FieldElm;

//     fn mul(self, rhs: &'a FieldElm) -> FieldElm {
//         &self * rhs
//     }
// }

// impl Mul for FieldElm {
//     type Output = FieldElm;

//     fn mul(self, rhs: FieldElm) -> FieldElm {
//         &self * &rhs
//     }
// }

// //
// // Addition trait implementation
// //

// impl Add for FieldElm {
//     type Output = FieldElm;

//     fn add(self, rhs: FieldElm) -> FieldElm {
//         FieldElm::from(self.value + rhs.value)
//     }
// }