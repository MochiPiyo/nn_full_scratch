//(data, label)
const NUMBER_OF_ROWS: usize = 28;
const NUMBER_OF_COLUMNS: usize = 28;
pub fn load_minst(image_path: &str, label_path: &str) -> (Vec<[[u8; NUMBER_OF_COLUMNS]; NUMBER_OF_ROWS]>, Vec<u8>) {
    println!("load_mnist");
    use std::fs::File;
    use std::io::Read;

    //load label
    let mut file = File::open(label_path).unwrap();
    let mut buf = Vec::new();
    let _ = file.read_to_end(&mut buf).unwrap();

    let mut labels = Vec::new();

    let mut offset: usize = 0;
    let _magic_number = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
    offset += 4;
    let number_of_items = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
    offset += 4;

    //label is array of u8
    for _i in 0..number_of_items {
        labels.push(buf[offset] as u8);
        offset += 1;
    }

    //load images
    let mut file = File::open(image_path).unwrap();
    let mut buf = Vec::new();
    let _ = file.read_to_end(&mut buf).unwrap();

    let mut offset: usize = 0;
    let _magic_number = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
    offset += 4;
    let number_of_images = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
    offset += 4;

    //28
    let number_of_rows = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
    offset += 4;
    //28
    let number_of_columns = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
    offset += 4;

    if number_of_rows as usize != NUMBER_OF_ROWS || number_of_columns as usize != NUMBER_OF_COLUMNS {
        panic!("load_minst: size err. readed data from file is : number_of_rows = {}, number_of_columns = {}", number_of_rows, number_of_columns);
    }else {
        println!("load_minst: number_of_images = {}, number_of_rows = {}, number_of_columns = {}",number_of_images, number_of_rows, number_of_columns);
    }

    //read pixel data
    let mut image_datas: Vec<[[u8; NUMBER_OF_COLUMNS]; NUMBER_OF_ROWS]> = vec![
        [[0; NUMBER_OF_COLUMNS]; NUMBER_OF_ROWS]; number_of_images as usize
    ];
    for image_id in 0..number_of_images {
        let image: &mut [[u8; NUMBER_OF_COLUMNS]; NUMBER_OF_ROWS] = &mut image_datas[image_id as usize];
        for row_id in 0..number_of_rows {
            for column_id in 0..number_of_columns {
                let this_pixel = buf[offset];
                offset += 1;

                image[row_id as usize][column_id as usize] = this_pixel;
            }
        }
    }

    if image_datas.len() == labels.len() {
        println!("num of data is '{}'", image_datas.len());
    }else {
        println!("err: image and label count is not same. image: {}, label: {}", image_datas.len(), labels.len());
        panic!();
    }

    return (image_datas, labels);
} 

pub fn selialize_minst(image_datas: &[[[u8; NUMBER_OF_COLUMNS]; NUMBER_OF_ROWS]]) -> Vec<[f32; 784]> {
    let mut returns: Vec<[f32; 784]> = Vec::with_capacity(image_datas.len());
    for image in image_datas.iter() {
        //28*28を784にする
        let mut selialized_image = [0.0; 784];
        let mut i = 0;
        for row in 0..NUMBER_OF_ROWS {
            for col in 0..NUMBER_OF_COLUMNS {
                selialized_image[i] = image[row][col] as f32;
                i += 1;
            }
        }
        
        returns.push(selialized_image)
    }
    return returns;
}