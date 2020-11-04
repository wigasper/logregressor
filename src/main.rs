use logregressor::model::*;

use std::fs::File;
use std::io::prelude::*;

type Element = f64;
type Matrix = (Vec<Element>, usize);

fn main() {
    let mut train_file = File::open("/home/wkg/repos/logregressor/mock_data/train").unwrap();
    let mut str_in = String::new();
    train_file.read_to_string(&mut str_in).expect("sdf");

    let mut x: Matrix = (Vec::new(), 0);
    let mut y: Matrix = (Vec::new(), 1);
    
    for line in str_in.split("\n") {
        if line.len() > 0 {
            let mut vals: Vec<&str> = line.split(",").collect();
            if x.1 == 0 {
                x.1 = vals.len() - 1;
            }

            let y_val = vals.pop().unwrap();
            y.0.push(y_val.parse::<f64>().unwrap_or_else(|why| {
                println!("{:?}", line);
                println!("{:?}", vals);
                panic!("could not parse {}: {}", y_val, why);
            }));
            for val in vals.iter() {
               x.0.push(val.parse::<f64>().unwrap()); 
            }
        }
    }
    
    let mut test_file = File::open("/home/wkg/repos/logregressor/mock_data/test").unwrap();
    str_in = String::new();
    test_file.read_to_string(&mut str_in).expect("sdf");

    let mut x_test: Matrix = (Vec::new(), 0);
    let mut y_test: Matrix = (Vec::new(), 1);
    
    for line in str_in.split("\n") {
        if line.len() > 0 {
            let mut vals: Vec<&str> = line.split(",").collect();
            if x_test.1 == 0 {
                x_test.1 = vals.len() - 1;
            }
            y_test.0.push(vals.pop().unwrap().parse::<f64>().unwrap());
            for val in vals.iter() {
               x_test.0.push(val.parse::<f64>().unwrap()); 
            }
        }
    }

    let mut model = LogRegressor::new();
    model.train(&x, &y, 5000, 0.1);
    model.test(&x_test, &y_test);
}
