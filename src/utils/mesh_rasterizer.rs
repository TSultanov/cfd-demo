use crate::quad_mesh::aabb::AABB;
use crate::quad_mesh::mesh::Mesh;
use crate::utils::drawing::draw_line;

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

/// Helper function that iterates over all cells in the mesh and draws their boundaries into the provided
/// pixel buffer. A linear mapping from coordinates in `bbox` to pixel space is computed, and the resulting
/// integers are used to draw the cell edges.
fn rasterize_mesh_impl(mesh: &Mesh, width: usize, height: usize, pixels_buffer: &mut [egui::Color32], bbox: AABB) {
    // Compute a common scale factor so that the full bbox fits into the image dimensions.
    let scale = ((width - 1) as f32 / bbox.width()).min((height - 1) as f32 / bbox.height());
    
    // These closures convert a coordinate in the mesh (or "polygon") space into a pixel coordinate.
    let poly_x_to_x = |x: f32| ((x - bbox.top_left().x) * scale).floor() as isize;
    let poly_y_to_y = |y: f32| ((y - bbox.top_left().y) * scale).floor() as isize;

    // Visit every cell in the mesh.
    mesh.visit_all_cells(|cell| {
        // For the current cell we compute its top left and bottom right coordinates.
        let half_width = cell.width / 2.0;
        let half_height = cell.height / 2.0;
        let poly_cell_top_left_x = cell.center.x - half_width;
        let poly_cell_top_left_y = cell.center.y - half_height;
        let poly_cell_bottom_right_x = cell.center.x + half_width;
        let poly_cell_bottom_right_y = cell.center.y + half_height;

        // Convert the cell boundary positions to their corresponding pixel coordinates.
        let x0 = poly_x_to_x(poly_cell_top_left_x);
        let y0 = poly_y_to_y(poly_cell_top_left_y);
        let x1 = poly_x_to_x(poly_cell_bottom_right_x);
        let y1 = poly_y_to_y(poly_cell_bottom_right_y);

        // Draw the four edges of the cell.
        draw_line(pixels_buffer, width, height, x0, y0, x1, y0, egui::Color32::BLACK);
        draw_line(pixels_buffer, width, height, x1, y0, x1, y1, egui::Color32::BLACK);
        draw_line(pixels_buffer, width, height, x1, y1, x0, y1, egui::Color32::BLACK);
        draw_line(pixels_buffer, width, height, x0, y1, x0, y0, egui::Color32::BLACK);
    });
}
