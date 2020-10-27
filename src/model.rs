
type Element = f64;
type Matrix = (Vec<Element>, usize);

#[derive(Default)]
pub struct LogRegressor {
    pub theta: Matrix;
    //pub learning_rate: f64;
    //pub n_iters: f64;
}

impl LogRegressor {
    pub fn train(X: &mut Matrix, y: &Matrix, n_iters: usize, 
                 mut learning_rate: f64) {
        // prepend column of 1s to X
        let mut ones: Matrix = (Vec::with_capacity(y.0.len()), 1);

        for _ in 0..y.0.len() {
            ones.0.push(1.0);
        }

        X = append_columns(&ones, &X);
        self.theta = zeros_matrix(X.1, 1);

        let mut losses: Vec<f64> = Vec::new();

        // gradient descent
        for _ in 0..n_iters {
            gd_step(X, y, learning_rate);
            losses.push(loss(X, y));
        }
    }
    
    pub fn test(X: &Matrix, y: &Matrix) {
        let y_hat = predict(X);
        let metrics = evaluate(&y_hat, y);

        println!("Accuracy: {}", metrics.0);
        println!("Precision: {}", metrics.1);
        println!("Recall: {}", metrics.2);
        println!("F1: {}", metrics.3);
    }

    pub fn predict(X: &Matrix) {
        let result = dot(X, self.theta);
        sigmoid(result);
        round(result);

        result
    }

    fn gd_step(X: &Matrix, y: &Matrix, learning_rate: f64) {
        let temp = tdot(X, sigmoid(dot(X, theta)));
        subtract(temp, y);
        multiply_scalar(temp, (learning_rate / y.0.len()));

        subtract(self.theta, temp);
    }

    fn loss(X: &Matrix, y: &Matrix) {
        let h: Matrix = sigmoid(dot(X, self.theta));
        let h_copy = h.copy();

        let epsilon = 0.00001;
        multiply_scalar(&mut y, -1.0);
        let term_0 = tdot(y, log_e_wise(h));

        multiply_scalar(&mut h_copy, -1);
        add_scalar(&mut h_copy, 1);
        log_e_wise(h_copy);
        let term_1 = tdot(add_scalar(y, 1.0), h_copy));
        
        let cost = (1.0 / y.0.len()) * (term_0 - term_1);
        
        cost
    }
}
