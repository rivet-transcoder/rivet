use super::super::framerate::estimate_frame_rate_from_ptses;

#[test]
fn estimate_frame_rate_from_uniform_ptses_returns_exact_fps() {
    // 24 fps: inter-PTS = 90000/24 = 3750 ticks.
    let ptses: Vec<i64> = (0..64).map(|i| i as i64 * 3750).collect();
    let fps = estimate_frame_rate_from_ptses(&ptses).expect("24 fps");
    assert!((fps - 24.0).abs() < 1e-9, "{} != 24.0", fps);
}

#[test]
fn estimate_frame_rate_from_reordered_ptses_sorts_before_delta() {
    // Same 24 fps, but decode-order != display-order (one B-frame
    // pair swapped). Median should still pick up the 3750-tick
    // period cleanly.
    let mut ptses: Vec<i64> = (0..32).map(|i| i as i64 * 3750).collect();
    ptses.swap(5, 6);
    ptses.swap(10, 11);
    let fps = estimate_frame_rate_from_ptses(&ptses).expect("24 fps after swap");
    assert!((fps - 24.0).abs() < 1e-9, "{} != 24.0", fps);
}

#[test]
fn estimate_frame_rate_from_single_outlier_delta_uses_median() {
    // 23 uniform 24-fps deltas + one 10× outlier. Median still 3750.
    let mut ptses: Vec<i64> = (0..24).map(|i| i as i64 * 3750).collect();
    ptses.push(24 * 3750 + 37500); // one huge gap
    let fps = estimate_frame_rate_from_ptses(&ptses).expect("24 fps despite outlier");
    assert!((fps - 24.0).abs() < 1e-9);
}

#[test]
fn estimate_frame_rate_returns_none_when_all_ptses_equal() {
    let ptses = vec![0i64; 10];
    assert!(estimate_frame_rate_from_ptses(&ptses).is_none());
}

#[test]
fn estimate_frame_rate_returns_none_when_fewer_than_two() {
    assert!(estimate_frame_rate_from_ptses(&[]).is_none());
    assert!(estimate_frame_rate_from_ptses(&[1234]).is_none());
}

#[test]
fn estimate_frame_rate_rejects_out_of_range_values() {
    // Single 1-tick delta → fps = 90000, outside [1, 240].
    let ptses = vec![0i64, 1];
    assert!(estimate_frame_rate_from_ptses(&ptses).is_none());
}

#[test]
fn estimate_frame_rate_handles_29_97_ntsc() {
    // 29.97 fps = 30000/1001. Inter-PTS = 90000 * 1001 / 30000 = 3003.
    let ptses: Vec<i64> = (0..32).map(|i| i as i64 * 3003).collect();
    let fps = estimate_frame_rate_from_ptses(&ptses).expect("29.97");
    assert!((fps - 30.0).abs() < 0.05, "got {}", fps); // 90000/3003 = 29.97..30.03
}
