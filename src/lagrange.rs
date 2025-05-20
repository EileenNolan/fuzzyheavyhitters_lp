use crate::field::FieldElm;
use std::ops::Mul;
use rand::Rng;
use std::collections::HashMap;
use std::convert::TryInto;
use crate::Group;
use std::ops::Neg;
use num_bigint::BigUint;


const D: usize = 2;
const DELTA: i64 = 3;
const P: i64 = 2;
const MODULUS_64: u64 = 9223372036854775783u64;
const STRING_LENGTH: usize = 4;

fn div_round_up(n: isize, d: isize) -> isize {
    (n + d - 1) / d
}

// Secret share y_values
fn secret_share_y_values(y_values: &[FieldElm]) -> (Vec<FieldElm>, Vec<FieldElm>) {
    let mut rng = rand::thread_rng();
    let mut y_A = Vec::new();
    let mut y_B = Vec::new();

    for y in y_values {
        let share_a = FieldElm::from(rng.gen_range(0, MODULUS_64));
       // let share_b = FieldElm::from((y.value + MODULUS_64 - share_a.value) % MODULUS_64);
        let share_b = FieldElm::from((y.value.clone() + MODULUS_64 - share_a.value.clone()) % MODULUS_64);


        y_A.push(share_a);
        y_B.push(share_b);
    }

    (y_A, y_B)
}

/// Compute polynomials and encode them
pub fn compute_polynomials(
    q: &[u64],
    d: usize,
    delta: i64,
) -> (Vec<Vec<FieldElm>>, Vec<Vec<FieldElm>>) {
    let mut E_A = Vec::new();
    let mut E_B = Vec::new();

    for i in 0..D {
        let mut x_values = Vec::new();
        let mut y_values_A = Vec::new();
        let mut y_values_B = Vec::new();

        for j in -DELTA..=DELTA {
            let key_j = FieldElm::from(q[i].wrapping_add(j as u64) % MODULUS_64);
            let dist = FieldElm::from((j.abs() as u64).pow(P as u32));

            // Secret share the distance value.
            // Ensure secret_share_y_values returns a tuple of two vectors; we then take the first share.
            let (dist_A, dist_B) = secret_share_y_values(&[dist]);
            // println!(
            //     "In compute_polynomials: dist_A = {:?} and dist_B = {:?}",
            //     dist_A, dist_B
            // );

            x_values.push(key_j.clone());
            y_values_A.push(dist_A[0].clone());
            y_values_B.push(dist_B[0].clone());
        }

        // Generate polynomial coefficients for each server.
        let poly_A = lagrange_interpolation_coeffs(&x_values, &y_values_A);
        let poly_B = lagrange_interpolation_coeffs(&x_values, &y_values_B);

        E_A.push(poly_A);
        E_B.push(poly_B);
    }

    (E_A, E_B)
}

//get the list of values required for each level of the tree
fn compute_list_tuples(q: isize, delta: isize) -> Vec<HashMap<isize, isize>> {
    //let limit = binary_tree_height(q.try_into().unwrap()); 
    let limit = STRING_LENGTH;
    let mut list_tuples: Vec<HashMap<isize, isize>> = Vec::new();

    for x in 0..=limit - 1 {
        let mut new_list_tuples: HashMap<isize, isize> = HashMap::new();

        if x == 0 {
            for j in -delta..=delta {
                let key = j + q;
                let val = j.abs();
                new_list_tuples.insert(key, val);
            }
        } else {
            if let Some(prev_map) = list_tuples.get(x - 1) {
                for (key, val) in prev_map.iter() {
                    let new_key = div_round_up(*key - 1, 2);
                    let new_val = *val;

                    new_list_tuples
                        .entry(new_key)
                        .and_modify(|existing_val| {
                            if new_val < *existing_val {
                                *existing_val = new_val;
                            }
                        })
                        .or_insert(new_val);
                }
            }
        }

        list_tuples.push(new_list_tuples);
    }

    list_tuples
}

pub fn compute_polynomials_prefix(target: &usize) -> (Vec<Vec<FieldElm>>, Vec<Vec<FieldElm>>) {
    let q: isize = *target as isize; // Dereference target, then cast
    let delta = DELTA;

    let list_tuples = compute_list_tuples(q, delta.try_into().unwrap());
    //let n = binary_tree_height(*target); 
    let n = STRING_LENGTH;
    println!("Height of tree is = {}", n);

    let mut E_A: Vec<Vec<FieldElm>> = Vec::new();
    let mut E_B: Vec<Vec<FieldElm>> = Vec::new();

    for y in 0..=n - 1 {
        let current_map = &list_tuples[y];
        let current_len = current_map.len();

        let max_allowed = if y == 0 {
            current_len
        } else {
            let value1 = 2usize.pow((n - y) as u32);
            let value2 = 2 + div_round_up((2 * delta - 1) as isize, 2);

            println!("Value1: {}", value1);
            println!("Value2: {}", value2);

            //let min_value = min(value1, value2);
            std::cmp::min(
               value1,
               value2 as usize
            )
        };

        // Prepare padded map
        let mut padded_map = current_map.clone();
        //if current_len < max_allowed {
        println!("Current n: {}", n);
        println!("Current length: {}", current_len);
        println!("Max allowed: {}", max_allowed);

        if current_len < max_allowed {
            //println!("Condition met: current_len < max_allowed");
    
            // let mut dummy_key = -(y as isize) - 1;
            // for _ in 0..(max_allowed - current_len) {
            //     while padded_map.contains_key(&dummy_key) {
            //         dummy_key -= 1;
            //     }
            //     padded_map.insert(dummy_key, 0);
            // }
            let mut rng = rand::thread_rng();
            let key_range = 0..MODULUS_64;
            let value_range = 0..MODULUS_64;
            
            for _ in 0..(max_allowed - current_len) {
                let mut dummy_key: isize;
                loop {
                    //dummy_key = rng.gen_range(key_range.clone()) as isize;
                    dummy_key = rng.gen_range(key_range.start, key_range.end) as isize;
                    if !padded_map.contains_key(&dummy_key) {
                        break;
                    }
                }
                //let dummy_value = rng.gen_range(value_range.clone());
                let dummy_value = rng.gen_range(value_range.start, value_range.end);
                padded_map.insert(dummy_key, dummy_value as isize);
            }
        }
        // Prepare x and y values
        let mut x_values: Vec<FieldElm> = Vec::new();
        let mut y_values: Vec<FieldElm> = Vec::new();

        for (x, y) in padded_map.iter() {
            x_values.push(FieldElm::from(*x as u64));
            y_values.push(FieldElm::from(*y as u64));
        }

        // Secret share y-values
        let (y_values_A, y_values_B) = secret_share_y_values(&y_values);

        println!("\nRound y = {}", y);
        println!("x_values: {:?}", x_values);
        println!("y_values (original): {:?}", y_values);

        // Compute polynomials
        let poly_A = lagrange_interpolation_coeffs(&x_values, &y_values_A);
        let poly_B = lagrange_interpolation_coeffs(&x_values, &y_values_B);

        E_A.push(poly_A);
        E_B.push(poly_B);
    }

    (E_A, E_B)
}

pub fn evaluate_client_polynomial(client_poly: &[Vec<FieldElm>], w: &[u64]) -> FieldElm {
    assert_eq!(client_poly.len(), w.len(), "Dimension mismatch.");
    //let mut sum = FieldElm::from(0);
    let mut sum = FieldElm::from(0u64);

    for i in 0..w.len() {
        // If no hashing is involved, directly convert:
        let key_i = FieldElm::from(w[i]);
        let x_i = evaluate_polynomial(&client_poly[i], &key_i);
        sum = FieldElm::from((sum.value + x_i.value) % MODULUS_64);
    }
    sum
}

/// Constructs the Lagrange basis polynomial coefficients for a given index `i`
pub fn lagrange_basis_coeffs(x_values: &[FieldElm], i: usize) -> Vec<FieldElm> {
    let mut coeffs = vec![FieldElm::one()];
    let mut denom = FieldElm::one();

    for (j, x_j) in x_values.iter().enumerate() {
        if i != j {
            let mut new_coeffs = vec![FieldElm::zero(); coeffs.len() + 1];

            for (k, c) in coeffs.iter().enumerate() {
                //new_coeffs[k] = new_coeffs[k].add(&c.negate().mul(x_j));
                let mut tmp = c.clone(); // Clone c if you need to keep its original value.
                tmp.negate();            // This negates tmp in-place.
                tmp.mul(x_j);            // This multiplies tmp in-place.
                new_coeffs[k].add(&tmp);
                
                //new_coeffs[k] = new_coeffs[k].add(&c.neg().mul(x_j));
                //new_coeffs[k + 1] = new_coeffs[k + 1].add(c);
                new_coeffs[k + 1].add(c);

            }

            coeffs = new_coeffs;
            //let diff = FieldElm::from((x_values[i].value + crate::field::MODULUS_64 - x_j.value) % crate::field::MODULUS_64);
            let diff = FieldElm::from((x_values[i].value.clone() + crate::field::MODULUS_64 - x_j.value.clone()) % crate::field::MODULUS_64);
            //denom = denom.mul(&diff);
            denom.mul(&diff);
        }
    }

    let denom_inv = denom.mod_inverse();
    //coeffs.iter().map(|c| c.mul(&denom_inv)).collect()
    coeffs.iter().map(|c| {
        let mut tmp = c.clone();
        tmp.mul(denom_inv.as_ref().expect("Expected a modular inverse"));
        tmp
    }).collect::<Vec<FieldElm>>()
    
}

/// Computes Lagrange interpolation coefficients for a given set of points
pub fn lagrange_interpolation_coeffs(x_values: &[FieldElm], y_values: &[FieldElm]) -> Vec<FieldElm> {
    assert_eq!(x_values.len(), y_values.len());

    let mut final_coeffs = vec![FieldElm::zero(); x_values.len()];

    for i in 0..x_values.len() {
        let basis_coeffs = lagrange_basis_coeffs(x_values, i);
        for (j, coeff) in basis_coeffs.iter().enumerate() {
            //final_coeffs[j] = final_coeffs[j].add(&coeff.mul(&y_values[i]));
            // Clone coeff so you don't modify the original if needed
            let mut tmp = coeff.clone();
            tmp.mul(&y_values[i]);      // This modifies tmp in place (tmp becomes coeff * y_values[i])
            final_coeffs[j].add(&tmp);   // This adds tmp to final_coeffs[j] in place

        }
    }

    final_coeffs
}

/// Evaluates a polynomial at a given field element
pub fn evaluate_polynomial(coeffs: &[FieldElm], x: &FieldElm) -> FieldElm {
    let mut result = FieldElm::zero();
    let mut power = FieldElm::one();

    for coeff in coeffs {
        //result = result.add(&coeff.mul(&power));
        // Clone coeff so as not to modify the original.
        let mut tmp = coeff.clone();
        // Multiply tmp in place.
        tmp.mul(&power);
        // Add the resulting value to result, in place.
        result.add(&tmp);

        //power = power.mul(x);
        power.mul(x);
    }

    result
}

/// Prints a polynomial in human-readable form
pub fn print_polynomial(coeffs: &[FieldElm], name: &str) {
    print!("{}(x) = ", name);
    let mut first = true;
    for (i, coeff) in coeffs.iter().enumerate().rev() {
        //if coeff.value != 0 {
        if coeff.value != BigUint::from(0u32) {
            if !first {
                print!(" + ");
            }
            print!("{}x^{}", coeff.value, i);
            first = false;
        }
    }
    println!();
}

// // tests to ensure that lagrange.rs is fully working
// // last two tests ensure that lagrange integrates with MPC
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::field::FieldElm;

//     // Test that everyone’s happy with secret sharing.
//     #[test]
//     fn test_secret_share_y_values() {
//         let y = FieldElm::from(12345);
//         let (share_a, share_b) = secret_share_y_values(&[y]);
//         let sum = (share_a[0].value + share_b[0].value) % MODULUS_64;
//         assert_eq!(sum, y.value, "Secret sharing reconstruction failed");
//     }

//     // Test Lagrange basis polynomial:
//     // For a given set of x_values, the basis polynomial for index i should evaluate
//     // to 1 at x_values[i] and 0 at x_values[j] for all j ≠ i.
//     #[test]
//     fn test_lagrange_basis_coeffs() {
//         // For a simple test, choose some small x-values.
//         let x_values: Vec<FieldElm> = vec![
//             FieldElm::from(1),
//             FieldElm::from(2),
//             FieldElm::from(3),
//         ];
//         // Test basis for index 0.
//         let basis = lagrange_basis_coeffs(&x_values, 0);
//         // Evaluate at each x and check the expected output.
//         for (i, x_i) in x_values.iter().enumerate() {
//             let eval = evaluate_polynomial(&basis, x_i);
//             if i == 0 {
//                 // Expect 1.
//                 assert_eq!(eval.value, FieldElm::one().value, "Basis at index 0 evaluated at x_0 should be 1, got {}", eval.value);
//             } else {
//                 // Expect 0.
//                 assert_eq!(eval.value, FieldElm::zero().value, "Basis at index 0 evaluated at x_{} should be 0, got {}", i, eval.value);
//             }
//         }
//     }

//     // Test full Lagrange interpolation:
//     // We set up known points such that f(x) = 2*x + 3.
//     // Our points: f(1)=5, f(2)=7, f(3)=9.
//     #[test]
//     fn test_lagrange_interpolation_coeffs() {
//         let x_values: Vec<FieldElm> = vec![
//             FieldElm::from(1),
//             FieldElm::from(2),
//             FieldElm::from(3),
//         ];
//         let y_values: Vec<FieldElm> = vec![
//             FieldElm::from(5),
//             FieldElm::from(7),
//             FieldElm::from(9),
//         ];
//         let poly = lagrange_interpolation_coeffs(&x_values, &y_values);
//         // Evaluate the polynomial at each x_value.
//         for (i, x_i) in x_values.iter().enumerate() {
//             let eval = evaluate_polynomial(&poly, x_i);
//             assert_eq!(
//                 eval.value, 
//                 y_values[i].value,
//                 "Interpolation at x_{} returned {} but expected {}",
//                 i,
//                 eval.value,
//                 y_values[i].value
//             );
//         }
//     }

//     // Test that evaluate_polynomial correctly computes the polynomial value.
//     #[test]
//     fn test_evaluate_polynomial() {
//         // Let’s define a simple polynomial: f(x) = 3 + 2*x.
//         let coeffs: Vec<FieldElm> = vec![FieldElm::from(3), FieldElm::from(2)]; // constant term, linear term
//         let x = FieldElm::from(10);
//         let result = evaluate_polynomial(&coeffs, &x);
//         // 3 + 2*10 = 23 (mod MODULUS_64)
//         assert_eq!(result.value, 23, "Expected 23, got {}", result.value);
//     }

//     // Test compute_polynomials:
//     // For each coordinate in q, the inner loop runs for j in [-DELTA, DELTA] and computes distances as |j|^P.
//     // For j = 0 the computed distance is (0^P)=0 so the secret share of 0 must be 0.
//     // Therefore, when evaluating the polynomial for a coordinate at x = FieldElm::from(q[i]),
//     // we expect the output to be 0.
//     #[test]
//     fn test_compute_polynomials_exact_match() {
//         // Use an input vector that exactly matches the server's dictionary.
//         let q: Vec<u64> = vec![5, 10, 15, 20];
//         let (E_A, E_B) = compute_polynomials(&q);
//         // Print polynomials for debugging
//         println!("Polynomials for Server A:");
//         for (i, poly) in E_A.iter().enumerate() {
//             let poly_name = format!("E_A{}", i);
//             print_polynomial(poly, &poly_name);
//         }
        
//         println!("Polynomials for Server B:");
//         for (i, poly) in E_B.iter().enumerate() {
//             let poly_name = format!("E_B{}", i);
//             print_polynomial(poly, &poly_name);
//         }
//         for i in 0..q.len() {
//             // The x_value corresponding to offset 0 is: FieldElm::from(q[i])
//             let x_eval = FieldElm::from(q[i]);
//             // Evaluate the derived polynomials (for both servers) at x_eval.
//             let eval_A = evaluate_polynomial(&E_A[i], &x_eval);
//             let eval_B = evaluate_polynomial(&E_B[i], &x_eval);
//             let combined = (eval_A.value + eval_B.value) % MODULUS_64;
//             assert_eq!(
//                 combined, 
//                 0,
//                 "For coordinate {}, combined evaluation of poly_A and poly_B at x = q[i] should yield 0, got {}",
//                 i, combined
//             );
//         }
//     }
    
//     #[test]
//     fn test_compute_polynomials_combined_outputs() {
//         // Use an input vector that exactly matches the server's dictionary.
//         let q: Vec<u64> = vec![5, 10, 15, 20];
//         let (E_A, E_B) = compute_polynomials(&q);

//         // Print the computed polynomials for each dimension for debugging.
//         println!("Polynomials for Server A:");
//         for (i, poly) in E_A.iter().enumerate() {
//             let poly_name = format!("E_A{}", i);
//             print_polynomial(poly, &poly_name);
//         }

//         println!("Polynomials for Server B:");
//         for (i, poly) in E_B.iter().enumerate() {
//             let poly_name = format!("E_B{}", i);
//             print_polynomial(poly, &poly_name);
//         }

//         // Sum the evaluated outputs for each server.
//         let mut sum_A: u64 = 0;
//         let mut sum_B: u64 = 0;

//         // Iterate through each coordinate.
//         for i in 0..q.len() {
//             // x_value corresponding to offset 0: exactly FieldElm::from(q[i])
//             let x_eval = FieldElm::from(q[i]);
//             // Evaluate the polynomial for both servers at x_eval.
//             let eval_A = evaluate_polynomial(&E_A[i], &x_eval);
//             let eval_B = evaluate_polynomial(&E_B[i], &x_eval);
//             println!(
//                 "Dimension {}: eval_A = {}, eval_B = {}",
//                 i, eval_A.value, eval_B.value
//             );

//             sum_A = (sum_A + eval_A.value) % MODULUS_64;
//             sum_B = (sum_B + eval_B.value) % MODULUS_64;
//         }

//         println!("Total sum for Server A evaluations = {}", sum_A);
//         println!("Total sum for Server B evaluations = {}", sum_B);

//         // Optionally, check that the combined result (sum_A + sum_B) modulo MODULUS_64 is 0.
//         let combined = (sum_A + sum_B) % MODULUS_64;
//         assert_eq!(combined, 0, "Combined evaluation sum should be 0, got {}", combined);
//     }

// }
