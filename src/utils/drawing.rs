
pub fn draw_line(
    pixels: &mut [egui::Color32],
    width: usize,
    height: usize,
    x0: isize,
    y0: isize,
    x1: isize,
    y1: isize,
    color: egui::Color32,
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy; // error term

    let mut x = x0;
    let mut y = y0;

    loop {
        // Set pixel if within bounds.
        if x >= 0 && x < width as isize && y >= 0 && y < height as isize {
            let idx = x as usize + y as usize * width;
            pixels[idx] = color;
        }

        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Draws a filled diamond centered at (cx, cy) in the pixel buffer.
/// The diamond is drawn in a 4x4 bounding box (approximately 4px wide) with the provided color.
pub fn draw_diamond(
    pixels: &mut [egui::Color32],
    width: usize,
    height: usize,
    cx: isize,
    cy: isize,
    color: egui::Color32,
) {
    // We'll draw the diamond in a 4x4 bounding box.
    let size: isize = 4;
    // Compute the "center" of the bounding box.
    // For an even size the center is ambiguous; here we use (size-1)/2 as a float.
    let center = (size as f32 - 1.0) / 2.0; // For size=4, center==1.5
    // Compute top-left of the bounding box (using floor so that the diamond is roughly centered).
    let top_left_x = (cx as f32 - center).floor() as isize;
    let top_left_y = (cy as f32 - center).floor() as isize;
    
    // We use a Manhattan distance condition (with a 0.5 tolerance)
    // so that the four edges are drawn.
    for j in 0..size {
        for i in 0..size {
            let local_x = i as f32;
            let local_y = j as f32;
            if (local_x - center).abs() + (local_y - center).abs() <= center + 0.5 {
                let x = top_left_x + i;
                let y = top_left_y + j;
                if x >= 0 && x < width as isize && y >= 0 && y < height as isize {
                    let idx = x as usize + y as usize * width;
                    pixels[idx] = color;
                }
            }
        }
    }
}
