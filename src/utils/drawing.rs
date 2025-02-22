
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