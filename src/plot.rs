

    pub fn plot_sample() {
        let mut window = kiss3d::window::Window::new("3D Scatterplot");
        
        // Example data points
        let points: Vec<kiss3d::nalgebra::Point3<f32>> = vec![
            kiss3d::nalgebra::Point3::new(0.0, 0.0, 0.0),
            kiss3d::nalgebra::Point3::new(1.0, 1.0, 1.0),
            kiss3d::nalgebra::Point3::new(2.0, -1.0, 0.5),
        ];
    
        // Render the points
        while window.render() {
            for point in &points {
                window.draw_point(point, &kiss3d::nalgebra::Point3::new(1.0, 0.0, 0.0)); // Red points
            }
        }
    }
    
