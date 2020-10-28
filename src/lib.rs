pub mod utils;
pub mod model;

#[cfg(test)]
mod tests {
    use crate::utils::*;

    type Element = f64;
    type Matrix = (Vec<Element>, usize);

    #[test]
    fn append_columns_0() {
        let a_vals = vec![1.2, 2.3, 3.4, 4.5, 5.6, 6.7];
        let a: Matrix = (a_vals, 3);

        let b_vals = vec![1.2, 2.3, 3.4, 4.5, 5.6, 6.7];
        let b: Matrix = (b_vals, 3);

        let result = append_columns(&a, &b);
        let e_vals = (vec![1.2, 2.3, 3.4, 1.2, 2.3, 3.4,
                           4.5, 5.6, 6.7, 4.5, 5.6, 6.7]);
        let expected = (e_vals, 6);

        assert_eq!(result.0, expected.0);
        assert_eq!(result.1, expected.1);
 	       
    }
    
    #[test]
    fn transpose_0() {
        let vals = vec![1.2, 2.3, 3.4, 4.5, 5.6, 6.7];
        let m: Matrix = (vals, 3);
        
        let result = transpose(&m);
        
        let e_vals = vec![1.2, 4.5, 2.3, 5.6, 3.4, 6.7];

        assert_eq!(result.1, 2);
        assert_eq!(result.0, e_vals);
    }
    
    #[test]
    fn dot_0() {
        let a_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a: Matrix = (a_vals, 3);

        let b_vals = vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0];
        let b: Matrix = (b_vals, 2);

        let result = dot(&a, &b);
        let e_vals = vec![31.0, 43.0,
                          67.0, 97.0];
        
        assert_eq!(result.0, e_vals);
    }
    
    #[test]
    #[should_panic]
    fn dot_1() {
        let a_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a: Matrix = (a_vals, 2);

        let b_vals = vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0];
        let b: Matrix = (b_vals, 2);

        let result = dot(&a, &b);

    }

    #[test]
    fn tdot_0() {
        let a_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a: Matrix = (a_vals, 3);

        let b_vals = vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0];
        let b: Matrix = (b_vals, 3);

        let result = tdot(&a, &b);
        let e_vals = vec![29.0, 34.0, 39.0,
                          37.0, 44.0, 51.0,
                          45.0, 54.0, 63.0];
        
        assert_eq!(result.0, e_vals);
    }

    #[test]
    fn mult_scalar_0() {
        let a_vals = vec![1.0, 2.0, 3.0, 4.0];
        let mut a: Matrix = (a_vals, 2);

        multiply_scalar(&mut a, 2.0);

        let e_vals = vec![2.0, 4.0, 6.0, 8.0];

        assert_eq!(a.0, e_vals);
    }
}
