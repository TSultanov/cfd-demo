use crate::quad_mesh::aabb::AABB;
use crate::quad_mesh::mesh::Mesh;
use crate::utils::drawing::{draw_line, draw_diamond};

/// Rasterizes a Mesh into an egui::ColorImage with no background. The image is created with the specified
/// width and height and all pixels are initialized as transparent before drawing the mesh cell boundaries.
/// The provided `bbox` is used to convert mesh-space coordinates into pixel coordinates.
pub fn rasterize_mesh_no_background(mesh: &Mesh, width: usize, height: usize, bbox: AABB) -> egui::ColorImage {
    let mut pixels = vec![egui::Color32::TRANSPARENT; width * height];
    rasterize_mesh_impl(mesh, width, height, &mut pixels, bbox);
    egui::ColorImage { size: [width, height], pixels }
}

/// Rasterizes a Mesh into an egui::ColorImage using an existing background image.
/// The provided `bbox` is used to convert mesh-space coordinates into pixel coordinates.
pub fn rasterize_mesh(mesh: &Mesh, background: egui::ColorImage, bbox: AABB) -> egui::ColorImage {
    let mut pixels = background.pixels;
    let width = background.size[0];
    let height = background.size[1];
    rasterize_mesh_impl(mesh, width, height, &mut pixels, bbox);
    egui::ColorImage { size: background.size, pixels }
}

/// Helper function that iterates over all cells in the mesh, draws their boundaries, and overlays the
/// intersection points as small (4px wide) yellow diamonds.
fn rasterize_mesh_impl(mesh: &Mesh, width: usize, height: usize, pixels_buffer: &mut [egui::Color32], bbox: AABB) {
    // Compute a common scale factor so that the full bbox fits into the image dimensions.
    let scale = ((width - 1) as f64 / bbox.width()).min((height - 1) as f64 / bbox.height());
    
    // These closures convert a coordinate in the mesh (or "polygon") space into a pixel coordinate.
    let poly_x_to_x = |x: f64| ((x - bbox.top_left().x) * scale).floor() as isize;
    let poly_y_to_y = |y: f64| ((y - bbox.top_left().y) * scale).floor() as isize;

    // Visit every cell in the mesh.
    mesh.visit_all_cells(|cell| {
        // Draw cell boundaries.
        let vertices = cell.quad.vertices();

        // Draw the cell outline by drawing a line between each consecutive pair of vertices.
        for i in 0..vertices.len() {
            let start = vertices[i];
            let end = vertices[(i + 1) % vertices.len()];
            let x0 = poly_x_to_x(start.x);
            let y0 = poly_y_to_y(start.y);
            let x1 = poly_x_to_x(end.x);
            let y1 = poly_y_to_y(end.y);
            draw_line(pixels_buffer, width, height, x0, y0, x1, y1, egui::Color32::BLACK);
        }
        
        // Draw a yellow diamond at each intersection point.
        for intersection in cell.intersections {
            let x = poly_x_to_x(intersection.x);
            let y = poly_y_to_y(intersection.y);
            draw_diamond(pixels_buffer, width, height, x, y, egui::Color32::ORANGE);
        }
    });
}
