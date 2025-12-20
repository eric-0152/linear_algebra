use crate::matrix::Matrix;
use crate::vector::Vector;
use std::fs::File;
use std::io::{BufRead, BufReader};

impl Vector {
    pub fn read_txt(path: &str) -> Result<Vector, String> {
        let file: Result<File, std::io::Error> = File::open(path);

        match file {
            Err(erroe_msg) => Err(erroe_msg.to_string()),
            Ok(file) => {
                let reader: BufReader<File> = BufReader::new(file);
                let mut vec: Vec<f64> = Vec::new();
                for line in reader.lines() {
                    let line = line.unwrap();
                    match line.trim().parse::<f64>() {
                        Err(error_msg) => return Err(error_msg.to_string()),
                        Ok(number) => vec.push(number),
                    }
                }

                Ok(Vector::from_vec(&vec))
            }
        }
    }
}

impl Matrix {
    pub fn read_txt(path: &str) -> Result<Matrix, String> {
        let file: Result<File, std::io::Error> = File::open(path);

        match file {
            Err(erroe_msg) => Err(erroe_msg.to_string()),
            Ok(file) => {
                let reader: BufReader<File> = BufReader::new(file);
                let mut rows: Vec<Vec<f64>> = Vec::new();
                for line in reader.lines() {
                    let line: String = line.unwrap();
                    let line: std::str::SplitWhitespace<'_> = line.split_whitespace();
                    let elements: Result<Vec<f64>, std::num::ParseFloatError> =
                        line.map(|number| number.parse::<f64>()).collect();
                    match elements {
                        Err(error_msg) => return Err(error_msg.to_string()),
                        Ok(numbers) => {
                            rows.push(numbers);
                        }
                    }
                }
                for r in 1..rows.len() {
                    if rows[0].len() != rows[r].len() {
                        return Err("Value Error: The size of rows are not match.".to_string());
                    }
                }

                Ok(Matrix::from_double_vec(&rows))
            }
        }
    }
}
