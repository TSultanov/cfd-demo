use crate::quad_mesh::polygon::Polygon;
use crate::utils::polygon_rasterizer::Rasterizer;
use std::rc::Rc;

pub struct MeshView {
    sketch: Rc<Polygon>,
    sketch_rasterizer: Rasterizer,
    sketch_texture: Option<egui::TextureHandle>,
}

impl Default for MeshView {
    fn default() -> Self {
        let polygon = Rc::new(default_polygon());
        let rasterizer = Rasterizer::new(polygon.clone());
        Self {
            sketch: polygon,
            sketch_texture: None,
            sketch_rasterizer: rasterizer,
        }
    }
}

impl eframe::App for MeshView {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_sketch(ui, ctx);
        });
    }
}

impl MeshView {
    fn draw_sketch(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let available_size = ui.available_rect_before_wrap().size();
        let bbox = self.sketch.bounding_box();
        let domain_aspect = bbox.width() / bbox.height();
        let available_aspect = available_size.x / available_size.y;
        let (img_width, img_height) = if available_aspect > domain_aspect {
            (available_size.y * domain_aspect, available_size.y)
        } else {
            (available_size.x, available_size.x / domain_aspect)
        };
        let img_size = egui::Vec2::new(img_width, img_height);

        let image = self
            .sketch_rasterizer
            .rasterize(img_width as usize, img_height as usize);

        let options = egui::TextureOptions {
            magnification: egui::TextureFilter::Nearest,
            minification: egui::TextureFilter::Nearest,
            mipmap_mode: Some(egui::TextureFilter::Nearest),
            wrap_mode: egui::TextureWrapMode::ClampToEdge,
        };

        if let Some(texture) = &mut self.sketch_texture {
            texture.set(image, options);
        } else {
            self.sketch_texture = Some(ctx.load_texture("sketch", image, options))
        }

        if let Some(texture) = &self.sketch_texture {
            ui.image((texture.id(), img_size));
        }
    }
}

fn default_polygon() -> Polygon {
    let mut poly = Polygon::new_rect(0.0, 0.0, 30.0, 10.0);

    poly.add_hole(Polygon::new_rect(7.0, 4.0, 2.0, 2.0))
        .unwrap();

    poly
}
