
type Element = f64;
type Matrix = (Vec<Element>, usize);

#[derive(Default)]
pub struct LogRegressor {
    pub theta: Vec<f64>;
    pub learning_rate: f64;
    pub n_iters: f64;
}

impl LogRegressor {
    pub fn train(X: &mut Matrix, y: &Matrix) {
        // prepend column of 1s to X
        let mut ones: Matrix = (Vec::with_capacity(y.0.len()), 1);

        for _ in 0..y.0.len() {
            ones.0.push(1.0);
        }

        X = append_columns(&ones, &X);
        self.theta = zeros_matrix(X.1, 1);
    }

    fn cost(X: &Matrix, y: &Matrix) {
        let h: Matrix = sigmoid(dot(X, self.theta));

        //cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))

        let epsilon = 0.00001;

        //let cost = (1 / y.0.len()) * 
    }
}
