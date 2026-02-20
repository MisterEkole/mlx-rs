use mlx::distributed;
use mlx::{Array, Device, DeviceType, Dtype};

fn main() {
    let cpu = Device::new(DeviceType::Cpu);
    cpu.set_default().unwrap();

    if !distributed::is_available() {
        println!("Distributed backend not available. Running single-process.\n");
    }

    let group = distributed::init(false);
    let rank = group.rank();
    let size = group.size();
    println!("[Rank {}/{}] Initialized", rank, size);

    // All Sum
    let x = Array::full(&[5], 1.0, Dtype::Float32).unwrap();
    match distributed::all_sum(&x, Some(&group)) {
        Ok(result) => println!("[Rank {}] all_sum = {:?}", rank, result),
        Err(e) => println!("[Rank {}] all_sum error: {}", rank, e),
    }

    // All Gather
    let my_data = Array::full(&[3], (rank + 1) as f32, Dtype::Float32).unwrap();
    match distributed::all_gather(&my_data, Some(&group)) {
        Ok(result) => println!("[Rank {}] all_gather = {:?}", rank, result),
        Err(e) => println!("[Rank {}] all_gather error: {}", rank, e),
    }

   
 
    if size >= 2 {
        if rank == 0 {
        let msg = Array::full(&[4], 42.0, Dtype::Float32).unwrap();
        match distributed::send(&msg, 1, Some(&group)) {
            Ok(dep) => {
                Array::eval(&dep).unwrap();
                println!("[Rank 0] Sent to rank 1");
            }
            Err(e) => println!("[Rank 0] Send error: {}", e),
        }
    } else if rank == 1 {
        let template = Array::zeros(&[4], Dtype::Float32).unwrap();
        match distributed::recv_like(&template, 0, Some(&group)) {
            Ok(received) => {
                Array::eval(&received).unwrap();
                println!("[Rank 1] Received: {:?}", received);
            }
            Err(e) => println!("[Rank 1] Recv error: {}", e),
        }
    }
}
    // Group Split
    if size > 1 {
        let subgroup = group.group_split(rank % 2, rank);
        println!(
            "[Rank {}] Subgroup: rank {} of {}",
            rank, subgroup.rank(), subgroup.size()
        );
    }

    println!("[Rank {}] Done!", rank);
}