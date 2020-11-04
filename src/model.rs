use crate::utils::*;

type Element = f64;
type Matrix = (Vec<Element>, usize);

#[derive(Default)]
pub struct LogRegressor {
    pub theta: Matrix,
}

// PMC2732298
//
impl LogRegressor {
    pub fn new() -> Self {
        Default::default()
    }

    fn gd_step(&mut self, x: &Matrix, y: &Matrix, learning_rate: f64) {
        let mut temp = dot(x, &self.theta);
        sigmoid(&mut temp);
        subtract(&mut temp, y);
        temp = tdot(x, &temp);
        
        multiply_scalar(&mut temp, learning_rate / y.0.len() as f64);

        subtract(&mut self.theta, &temp);
    }
 
    pub fn predict(&self, x: &Matrix) -> Matrix {
        let mut result = dot(x, &self.theta);
        sigmoid(&mut result);
        round(&mut result);

        result
    }


    pub fn loss(&self, x: &Matrix, y: &Matrix) -> f64 {
        let mut h: Matrix = dot(x, &self.theta);
        sigmoid(&mut h);
        
        let mut h_copy = h.to_owned();
        let mut y_copy = y.to_owned();

        multiply_scalar(&mut y_copy, -1.0);
        log_e_wise(&mut h);
        let term_0 = tdot(y, &h);

        multiply_scalar(&mut h_copy, -1.0);
        add_scalar(&mut h_copy, 1.0);
        log_e_wise(&mut h_copy);
        add_scalar(&mut y_copy, 1.0);
        let term_1 = tdot(&y_copy, &h_copy);
        
        // TODO make this nicer:
        if term_0.0.len() != 1 || term_1.0.len() != 1 {
            panic!("model::loss - term_0 or term_1 len is not 1");
        }

        (1.0 / y.0.len() as f64) * (term_0.0.get(0).unwrap() - term_1.0.get(0).unwrap())
    }  

    pub fn train(&mut self, x_in: &Matrix, y: &Matrix, n_iters: usize, 
                 learning_rate: f64) {
        // prepend column of 1s to X
        let mut ones: Matrix = (Vec::with_capacity(y.0.len()), 1);

        for _ in 0..y.0.len() {
            ones.0.push(1.0);
        }

        let x: Matrix = append_columns(&ones, &x_in);
        self.theta = zeros_matrix(x.1, 1);

        let mut losses: Vec<f64> = Vec::new();

        // gradient descent
        for _ in 0..n_iters {
            self.gd_step(&x, y, learning_rate);
            losses.push(self.loss(&x, y));
        }
    }
    
    pub fn test(&self, x_in: &Matrix, y: &Matrix) {
        // prepend column of 1s to X
        let mut ones: Matrix = (Vec::with_capacity(y.0.len()), 1);

        for _ in 0..y.0.len() {
            ones.0.push(1.0);
        }

        let x: Matrix = append_columns(&ones, &x_in);

        let y_hat: Matrix = self.predict(&x);
        let metrics = evaluate(&y_hat, y);

        println!("Accuracy: {}", metrics.0);
        println!("Precision: {}", metrics.1);
        println!("Recall: {}", metrics.2);
        println!("F1: {}", metrics.3);
    }

}
