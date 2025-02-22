use crate::quad_mesh::mesh::{tesselate, Cell};
use crate::quad_mesh::point::Point;
use crate::quad_mesh::polygon::Polygon;
use crate::utils::polygon_rasterizer::PolygonRasterizer;
use std::rc::Rc;
use crate::quad_mesh::aabb::AABB;
use crate::utils::mesh_rasterizer::{rasterize_mesh, rasterize_mesh_no_background};

pub struct MeshParams {
    pub feature_size: f32,
    pub max_cell_size: f32,
}

impl Default for MeshParams {
    fn default() -> Self {
        Self {
            feature_size: 0.1,
            max_cell_size: 0.5,
        }
    }
}

pub struct MeshView {
    sketch: Rc<Polygon>,
    sketch_rasterizer: PolygonRasterizer,
    sketch_texture: Option<egui::TextureHandle>,
    mesh_texture: Option<egui::TextureHandle>,

    mesh: Option<Cell>,
    mesh_params: MeshParams,
}

impl Default for MeshView {
    fn default() -> Self {
        let polygon = Rc::new(default_polygon());
        let rasterizer = PolygonRasterizer::new(polygon.clone());
        Self {
            sketch: polygon,
            sketch_texture: None,
            sketch_rasterizer: rasterizer,
            mesh_texture: None,
            mesh: None,
            mesh_params: MeshParams::default(),
        }
    }
}

impl eframe::App for MeshView {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("mesh_params").show(ctx, |ui| {
            ui.heading("Mesh Parameters");
            ui.label("Feature Size");
            ui.add(egui::Slider::new(&mut self.mesh_params.feature_size, 0.01..=0.5));
            ui.label("Max Cell Size");
            ui.add(egui::Slider::new(&mut self.mesh_params.max_cell_size, 0.1..=1.0));
            if ui.button("Tesselate").clicked() {
                self.mesh = Some(tesselate(&self.sketch, self.mesh_params.feature_size, self.mesh_params.max_cell_size));
            }
        });

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

        let image = if let Some(mesh) = &self.mesh {
            rasterize_mesh(mesh, image, self.sketch.bounding_box())
        } else {
            image
        };

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

        if let(Some(mesh)) = &self.mesh {
            let available_size = ui.available_rect_before_wrap().size();
            let bbox = mesh.boundary;
            let domain_aspect = bbox.width() / bbox.height();
            let available_aspect = available_size.x / available_size.y;
            let (img_width, img_height) = if available_aspect > domain_aspect {
                (available_size.y * domain_aspect, available_size.y)
            } else {
                (available_size.x, available_size.x / domain_aspect)
            };
            let img_size = egui::Vec2::new(img_width, img_height);

            let mesh_image = rasterize_mesh_no_background(mesh, img_width as usize, img_height as usize);

            if let Some(texture) = &mut self.mesh_texture {
                texture.set(mesh_image, options);
            } else {
                self.mesh_texture = Some(ctx.load_texture("mesh", mesh_image, options));
            };

            if let Some(mesh_texture) = &self.mesh_texture {
                ui.image((mesh_texture.id(), img_size));
            }
        }
    }
}

fn default_polygon() -> Polygon {
    let mut poly = Polygon::new_rect(0.0, 0.0, 30.0, 10.0);

    poly.add_hole(Polygon::new_polygon(
        Point { x: 5.0, y: 5.0 },
        1.0,
        4,
        std::f32::consts::TAU / 8.0,
    ))
    .unwrap();

    poly
}
