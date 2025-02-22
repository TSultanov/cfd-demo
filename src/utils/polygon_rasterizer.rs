use crate::quad_mesh::point::Point;
use crate::quad_mesh::polygon::Polygon;
use std::rc::Rc;
use crate::utils::drawing::draw_line;

pub struct PolygonRasterizer {
    polygon: Rc<Polygon>,
    cache: Option<egui::ColorImage>,
    cached_size: Option<(usize, usize)>,
}

impl PolygonRasterizer {
    pub fn new(polygon: Rc<Polygon>) -> Self {
        Self {
            polygon,
            cache: None,
            cached_size: None,
        }
    }

    pub fn rasterize(&mut self, width: usize, height: usize) -> egui::ColorImage {
        if let Some((cached_width, cached_height)) = self.cached_size {
            if cached_width == width && cached_height == height {
                if let Some(cache) = &self.cache {
                    return cache.clone();
                }
            }
        }

        let mut pixels = vec![egui::Color32::TRANSPARENT; width * height];
        rasterize_polygon(&self.polygon, width, height, &mut pixels);
        let image = egui::ColorImage {
            size: [width, height],
            pixels,
        };
        self.cache = Some(image.clone());
        self.cached_size = Some((width, height));

        image
    }
}

fn rasterize_polygon(
    sketch: &Polygon,
    width: usize,
    height: usize,
    pixels_buffer: &mut [egui::Color32],
) {
    let bbox = sketch.bounding_box();

    let scale = ((width - 1) as f32 / bbox.width()).min((height - 1) as f32 / bbox.height());

    let x_to_poly_x = |x: usize| x as f32 / scale + bbox.top_left().x;
    let y_to_poly_y = |y: usize| y as f32 / scale + bbox.top_left().y;
    let poly_x_to_x = |x: f32| ((x - bbox.top_left().x) * scale).floor() as usize;
    let poly_y_to_y = |y: f32| ((y - bbox.top_left().y) * scale).floor() as usize;

    for y in 0..height {
        for x in 0..width {
            let idx = x + y * width;
            let p = Point {
                x: x_to_poly_x(x),
                y: y_to_poly_y(y),
            };
            pixels_buffer[idx] = if sketch.contains_point(&p) {
                egui::Color32::LIGHT_BLUE
            } else {
                egui::Color32::TRANSPARENT
            }
        }
    }

    let mut draw_edge = |edge: (Point, Point)| {
        let p0 = edge.0;
        let p1 = edge.1;
        let x0 = poly_x_to_x(p0.x);
        let y0 = poly_y_to_y(p0.y);
        let x1 = poly_x_to_x(p1.x);
        let y1 = poly_y_to_y(p1.y);
        draw_line(
            pixels_buffer,
            width,
            height,
            x0 as isize,
            y0 as isize,
            x1 as isize,
            y1 as isize,
            egui::Color32::BLACK,
        );
    };

    // Draw edges
    for edge in sketch.edges() {
        draw_edge(edge);
    }

    for hole in &sketch.holes {
        for edge in hole.edges() {
            draw_edge(edge);
        }
    }
}
