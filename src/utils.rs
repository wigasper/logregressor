
type Element = f64;
type Matrix = (Vec<Element>, usize);

pub fn transpose(m: &Matrix) -> Matrix {
	let mut m_out: Matrix = (Vec::new(), m.0.len() / m.1);

    for col in 0..m.1 {
    	for row in 0..(m.0.len() / m.1) {
        	m_out.0.push(m.0.get(row * m.1 + col).unwrap().to_owned());
        }
    }

	m_out
}

pub fn dot(a: &Matrix, b: &Matrix) -> Matrix {
	let mut m_out: Matrix = (Vec::with_capacity(b.1 * a.0.len() / a.1), b.1);
	
	if a.1 != (b.0.len() / b.1) {
		panic!("utils::dot - matrices are not conformable!");
	}

	for row in 0..(a.0.len() / a.1) {
		for col in 0..(b.1) {
			let mut sum: Element = 0.0;
			
			let a_begin: usize = row * a.1;
			let a_end: usize = (row * a.1) + a.1;
			
			let mut b_idx = col;
			
			for a_idx in a_begin..a_end {
				sum += a.0.get(a_idx).unwrap() * b.0.get(b_idx).unwrap();
				b_idx += b.1;
			}
			
			m_out.0.push(sum);
		}
	}	
    m_out
}

// A special dot function, multiplies the transpose of the first matrix a by
// the second matrix b. Avoids a memory allocation
pub fn tdot(a: &Matrix, b: &Matrix) -> Matrix {
 	let mut m_out: Matrix = (Vec::with_capacity(b.1 * a.0.len() / a.1), b.1);
	
    if a.0.len() / a.1 != (b.0.len() / b.1) {
		panic!("utils::tdot - matrices are not conformable!");
	}

	for a_col in 0..a.1 {
		for col in 0..b.1 {
			let mut sum: Element = 0.0;
			
			let mut a_idx: usize = a_col;
			//let a_end: usize = a_col;
			
			let mut b_idx: usize = col;
			
			for _ in 0..(a.0.len() / a.1) {
				sum += a.0.get(a_idx).unwrap() * b.0.get(b_idx).unwrap();
				a_idx += a.1;
                b_idx += b.1;
			}
			
			m_out.0.push(sum);
		}
	}	
    m_out   
}

pub fn zeros_matrix(m: usize, n: usize) -> Matrix {
    let mut m_out: Matrix = (Vec::with_capacity(m * n), n);

    for _ in 0..(m * n) {
        m_out.0.push(0.0);
    }

    m_out
}


// append b columns to a
pub fn append_columns (a: &Matrix, b: &Matrix) -> Matrix {
    let a_n_rows: usize = a.0.len() / a.1;
    let b_n_rows: usize = b.0.len() / b.1;

    if a_n_rows != b_n_rows {
        panic!("utils::append_columns - matrices do not have same number of rows!");
    }
	
    let mut m_out: Matrix = (Vec::with_capacity(a.0.len() + b.0.len()), a.1 + b.1);

    for row in 0..a_n_rows {
        let a_start_idx: usize = row * a.1;
        for a_idx in a_start_idx..(row * a.1 + a.1) {
            m_out.0.push(a.0.get(a_idx).unwrap().to_owned());
        }

        let b_start_idx: usize = row * b.1;
        for b_idx in b_start_idx..(row * b.1 + b.1) {
            m_out.0.push(b.0.get(b_idx).unwrap().to_owned());
        }
    }

    m_out
}

pub fn sigmoid(m: &mut Matrix) { 
    for val in m.0.iter_mut() {
        *val = 1.0 / (1.0 + (-1.0 * *val).exp());
    }
}

pub fn multiply_scalar(m: &mut Matrix, scalar: f64) {
    for val in m.0.iter_mut() {
        *val = scalar * *val;
    }
}
