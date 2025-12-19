use crate::matrix::Matrix;
use crate::solve;
use crate::vector::Vector;
// use crate::decomposition;

pub fn rayleigh_quotient(eigenvector: &Vector, matrix_a: &Matrix) -> f64 {
    let numerator: f64 = eigenvector
        .transpose()
        .multiply_Vector(&matrix_a.multiply_Vector(eigenvector).unwrap())
        .unwrap()
        .entries[0];
    let denominator: f64 = eigenvector
        .transpose()
        .multiply_Vector(eigenvector)
        .unwrap()
        .entries[0];

    numerator / denominator
}

/// Return a similar matrix.
///
/// ### Algorithm :
/// ***A = QR*** => let ***S = RQ*** => ***IS = Q^(-1)QRQ*** => ***S = Q^(-1)AQ***
pub fn similar_matrix(matrix: &Matrix) -> Result<Matrix, String> {
    if matrix.row != matrix.col {
        return Err("Input Error: The matrix is not square.".to_string());
    }
    
    match matrix.qr_decomposition() {
        Ok(tuple) => {
            Ok(tuple.1.multiply_Matrix(&tuple.0).unwrap())
        }
        Err(error_msg) => Err(error_msg),
    }
}

pub fn shift_qr_algorithm(
    matrix: &Matrix,
    max_iter: u32,
    error_thershold: f64,
) -> Result<(Matrix, f64), String> {
    match similar_matrix(matrix) {
        Ok(mut matrix_s) => {
            let last_row_idx: usize = matrix_s.row - 1;
            let last_col_idx: usize = matrix_s.col - 1;
            let matrix_size: usize = matrix_s.row;
            let mut last_eigenvalue: f64 = matrix_s.entries[last_row_idx][last_col_idx];
            let mut difference: f64 = 1.0;
            let mut step: u32 = 0;

            while difference > error_thershold && step < max_iter {
                let shift = Matrix::identity(matrix_size).multiply_scalar(&last_eigenvalue);
                matrix_s = matrix_s.substract_Matrix(&shift).unwrap();
                matrix_s = similar_matrix(&matrix_s)
                    .unwrap()
                    .add_Matrix(&shift)
                    .unwrap();
                difference = (matrix_s.entries[last_row_idx][last_col_idx] - last_eigenvalue).abs();
                last_eigenvalue = matrix_s.entries[last_row_idx][last_col_idx];
                step += 1;
            }

            Ok((matrix_s, difference))
        }

        Err(error_msg) => Err(error_msg),
    }
}

/// ## Can not return complex eigenvalue
/// Return a vector which contains eigenvalue of matrix and the difference.
///
/// Check the difference, if it's too large, the eigenvalue may contains complex number.
pub fn eigenvalue_with_qr(
    matrix: &Matrix,
    max_iter: u32,
    error_thershold: f64,
) -> Result<(Vector, f64), String> {
    match similar_matrix(matrix) {
        Ok(_) => match shift_qr_algorithm(matrix, max_iter, error_thershold) {
            Ok(tuple) => {
                let matrix_s: Matrix = tuple.0;
                let difference: f64 = tuple.1;
                let mut eigenvalue: Vec<f64> = Vec::new();
                for d in 0..matrix_s.row {
                    eigenvalue.push(matrix_s.entries[d][d]);
                }

                Ok((Vector::from_vec(&eigenvalue), difference))
            }

            Err(error_msg) => Err(error_msg),
        },

        Err(error_msg) => Err(error_msg),
    }
}

/// ## Can not return complex eigenvector
pub fn eigenvector_wirh_qr(
    matrix: &Matrix,
    max_iter: u32,
    error_thershold: f64,
) -> Result<((Vector, f64)), String> {
    match shift_qr_algorithm(matrix, max_iter, error_thershold) {
        Ok(tuple) => {
            let matrix_s: Matrix = tuple.0;
            let difference: f64 = tuple.1;

            matrix_s.display();
            let mut modified_matrix = matrix.clone();
            for d in 0..modified_matrix.row {
                modified_matrix.entries[d][d] -= matrix_s.entries[0][0];
            }

            match solve::gauss_jordan_elimination(
                &modified_matrix,
                &Vector::zeros(modified_matrix.col),
            ) {
                Ok(tuple) => {
                    match solve::upper_triangular(&tuple.0, &Vector::zeros(matrix_s.col)) {
                        Ok(eigen_vector) => Ok((
                            eigen_vector.swap_with_permutation(&tuple.2).unwrap(),
                            difference,
                        )),
                        Err(error_msg) => Err(error_msg),
                    }
                }

                Err(error_msg) => Err(error_msg),
            }
        }

        Err(error_msg) => Err(error_msg),
    }
}
