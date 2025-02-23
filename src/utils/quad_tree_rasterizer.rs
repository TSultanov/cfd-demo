use std::collections::VecDeque;
use crate::quad_mesh::aabb::AABB;
use crate::quad_mesh::quad_tree::QuadTree;
use crate::utils::drawing::draw_line;

pub fn rasterize_quad_tree_no_background(mesh: &QuadTree, width: usize, height: usize) -> egui::ColorImage {
    let mut pixels = vec![egui::Color32::TRANSPARENT; width * height];
    rasterize_quad_tree_impl(mesh, width, height, &mut pixels, mesh.boundary);
    egui::ColorImage {
        size: [width, height],
        pixels,
    }
}

pub fn rasterize_quad_tree(mesh: &QuadTree, background: egui::ColorImage, bbox: AABB) -> egui::ColorImage {
    let mut pixels = background.pixels;
    rasterize_quad_tree_impl(mesh, background.size[0], background.size[1], &mut pixels, bbox);
    egui::ColorImage {
        size: background.size,
        pixels,
    }
}

fn rasterize_quad_tree_impl(
    cell: &QuadTree,
    width: usize,
    height: usize,
    pixels_buffer: &mut [egui::Color32],
    bbox: AABB,
) {
    let mut queue = VecDeque::from(vec![cell]);

    let scale = ((width - 1) as f32 / bbox.width()).min((height - 1) as f32 / bbox.height());

    // let x_to_poly_x = |x: usize| x as f32 / scale + bbox.top_left().x;
    // let y_to_poly_y = |y: usize| y as f32 / scale + bbox.top_left().y;
    let poly_x_to_x = |x: f32| ((x - bbox.top_left().x) * scale).floor() as usize;
    let poly_y_to_y = |y: f32| ((y - bbox.top_left().y) * scale).floor() as usize;

    while let Some(cell) = queue.pop_front() {
        if cell.is_leaf() {
            let boundary = cell.boundary;
            let x0 = poly_x_to_x(boundary.top_left().x) as isize;
            let y0 = poly_y_to_y(boundary.top_left().y) as isize;
            let x1 = poly_x_to_x(boundary.bottom_right().x) as isize;
            let y1 = poly_y_to_y(boundary.bottom_right().y) as isize;
            draw_line(pixels_buffer, width, height, x0, y0, x1, y0, egui::Color32::BLACK);
            draw_line(pixels_buffer, width, height, x1, y0, x1, y1, egui::Color32::BLACK);
            draw_line(pixels_buffer, width, height, x1, y1, x0, y1, egui::Color32::BLACK);
            draw_line(pixels_buffer, width, height, x0, y1, x0, y0, egui::Color32::BLACK);
        } else {
            if let Some(children) = &cell.children {
                for child in children.iter() {
                    queue.push_back(child);
                }
            }
        }
    }
}