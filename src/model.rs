use crate::utils::*;

type Element = f64;
type Matrix = (Vec<Element>, usize);

#[derive(Default)]
pub struct LogRegressor {
    pub theta: Matrix,
    //pub learning_rate: f64;
    //pub n_iters: f64;
}

impl LogRegressor {
    pub fn new() -> Self {
        Default::default()
    }

    fn gd_step(&self, X: &Matrix, y: &Matrix, learning_rate: f64) {
        let mut temp = dot(X, &self.theta);
        sigmoid(&mut temp);
        temp = tdot(X, &temp);
        //let mut temp = tdot(X, &sigmoid(&dot(X, self.theta)));
        subtract(&mut temp, y);
        multiply_scalar(&mut temp, learning_rate / y.0.len() as f64);

        subtract(&mut self.theta, &temp);
    }
 
    pub fn predict(&self, X: &Matrix) -> Matrix {
        let result = dot(X, &self.theta);
        sigmoid(&mut result);
        round(&mut result);

        result
    }


    fn loss(&self, X: &Matrix, y: &Matrix) -> f64 {
        let mut h: Matrix = dot(X, &self.theta);
        sigmoid(&mut h);
        //let h: Matrix = sigmoid(&mut dot(X, self.theta));
        let h_copy = h.to_owned();

        let epsilon = 0.00001;
        multiply_scalar(&mut y, -1.0);
        log_e_wise(&mut h);
        let term_0 = tdot(y, &h);

        multiply_scalar(&mut h_copy, -1.0);
        add_scalar(&mut h_copy, 1.0);
        log_e_wise(&mut h_copy);
        add_scalar(&mut y, 1.0);
        let term_1 = tdot(y, &h_copy);
        
        let cost = (1.0 / y.0.len() as f64) * (term_0.0.get(0).unwrap() - term_1.0.get(0).unwrap());
        
        cost
    }  

    pub fn train(&self, X_in: &Matrix, y: &Matrix, n_iters: usize, 
                 mut learning_rate: f64) {
        // prepend column of 1s to X
        let mut ones: Matrix = (Vec::with_capacity(y.0.len()), 1);

        for _ in 0..y.0.len() {
            ones.0.push(1.0);
        }

        let X: Matrix = append_columns(&ones, &X_in);
        self.theta = zeros_matrix(X.1, 1);

        let mut losses: Vec<f64> = Vec::new();

        // gradient descent
        for _ in 0..n_iters {
            self.gd_step(&X, y, learning_rate);
            losses.push(self.loss(&X, y));
        }
    }
    
    pub fn test(&self, X: &Matrix, y: &Matrix) {
        let y_hat: Matrix = self.predict(X);
        let metrics = evaluate(&y_hat, y);

        println!("Accuracy: {}", metrics.0);
        println!("Precision: {}", metrics.1);
        println!("Recall: {}", metrics.2);
        println!("F1: {}", metrics.3);
    }

}
