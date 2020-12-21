#include <iostream>
#include <mitsuba/python/python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <time.h>

class HeightMap {
    using Image = py::array_t<int32_t>;
    using array_f = std::vector<float>;

public:
    enum Interpolation{
        NEAREST = 0
    };

    HeightMap(std::vector<Image> map_list, ssize_t im_size, array_f x_range,
              array_f y_range, array_f sigma_n, array_f x_min, array_f y_max,
              Interpolation interpolation=NEAREST){
        m_data = map_list;
        m_im_size = im_size;
        m_x_range = x_range;
        m_y_range = y_range;
        m_sigma_n = sigma_n;
        m_x_min = x_min;
        m_y_max = y_max;
        m_shape_image = std::vector<ssize_t>{im_size, im_size};
        m_interpolation = interpolation;
        m_r_sqr = (im_size / 2) * (im_size / 2);
        m_center_uv = im_size / 2;
    }

    HeightMap(ssize_t im_size, Interpolation interpolation=NEAREST){
        m_shape_image = std::vector<ssize_t>{im_size, im_size};
        m_interpolation = interpolation;
        m_im_size = im_size;
        m_r_sqr = (im_size / 2) * (im_size / 2);
        m_center_uv = im_size / 2;
    }

    ~HeightMap(){};

    auto get_height_map(py::array_t<float> in_pos, py::array_t<int32_t> mesh_id){

        clock_t start = clock();
        std::cout << "get_map_start" << std::endl;

        ssize_t n_sample = mesh_id.size();

        std::vector<ssize_t> shape_result{n_sample, 1, m_im_size, m_im_size};

        Image result{shape_result};

        // Roop of sampling for each ray
        for (int i = 0; i < n_sample; i++){
            int32_t id_i = *mesh_id.data(i);
            if(id_i == 0){
                continue;
            }

            id_i -= 1;

            clip_scaled_map(result, i,  m_data[id_i], *in_pos.data(i, 0), *in_pos.data(i, 1),
                            m_sigma_n[id_i], m_x_range[id_i], m_y_range[id_i],
                            m_x_min[id_i], m_y_max[id_i]);
        }

        std::cout << "get_map_end (took " << (double)(clock() - start) / CLOCKS_PER_SEC << " s)" << std::endl;
        return result;
    }


    Image clip_scaled_map(Image map_scaled, float x_in, float y_in, float sigma_n,
                          float x_range, float y_range, float x_min, float y_max){
        
        Image map_cliped{m_shape_image};
        const auto map_buf = map_scaled.request();
        const auto map_shape = map_buf.shape;

        int height = map_shape[0];
        int width = map_shape[1];

        // Length of a pixel edge in map_scaled
        float px_len = x_range / float(width);

        // The number of pixels in the range of 12 sigma_n
        float scale_px = 12 * sigma_n / px_len;

        // Ratio of pixel between map_scaled and map_cliped
        // This means the number of pixels of map_scaled in a pixel of map_cliped 
        float ratio_px = scale_px / (float)m_im_size;

        // uv position of center of map_cliped
        float u_c = (y_max - y_in) / y_range * height;
        float v_c = (x_in - x_min) / x_range * width;

        // roop of u (= y)
        for (int i = 0; i < m_im_size; i++){
            int dist_u = i - m_center_uv;
            // u position of a pixel of interested in map_scaled
            float px_u = u_c + dist_u * ratio_px;

            // roop of v (= x) 
            for(int j = 0; j < m_im_size; j++){
                int dist_v = j - m_center_uv;
                // v position of a pixel of interested in map_scaled
                float px_v = v_c + dist_v * ratio_px;

                // if the pixel is out of range 6 sigma_n, fill 0
                if (dist_u * dist_u + dist_v * dist_v > m_r_sqr){
                    *map_cliped.mutable_data(i, j) = 0;
                }else{
                    if (px_u >= 0 && px_v >= 0 && px_u < height && px_v < width){
                        *map_cliped.mutable_data(i, j) = pick_pxl(map_scaled, px_u,
                                                                    px_v, m_interpolation);
                    }else{
                        // if the pixel if out of map_scaled, fill 31
                        *map_cliped.mutable_data(i, j) = 31;
                    }
                }
            }
        }

        return map_cliped;
    }

    void clip_scaled_map(Image &map_list, int num ,Image map_scaled, float x_in, float y_in, float sigma_n,
                         float x_range, float y_range, float x_min, float y_max){
        
        Image map_cliped{m_shape_image};
        const auto map_buf = map_scaled.request();
        const auto map_shape = map_buf.shape;

        int height = map_shape[0];
        int width = map_shape[1];

        // Length of a pixel edge in map_scaled
        float px_len = x_range / float(width);

        // The number of pixels in the range of 12 sigma_n
        float scale_px = 12 * sigma_n / px_len;

        // Ratio of pixel between map_scaled and map_cliped
        // This means the number of pixels of map_scaled in a pixel of map_cliped 
        float ratio_px = scale_px / (float)m_im_size;

        // uv position of center of map_cliped
        float u_c = (y_max - y_in) / y_range * height;
        float v_c = (x_in - x_min) / x_range * width;

        // roop of u (= y)
        for (int i = 0; i < m_im_size; i++){
            int dist_u = i - m_center_uv;
            // u position of a pixel of interested in map_scaled
            float px_u = u_c + dist_u * ratio_px;

            // roop of v (= x) 
            for(int j = 0; j < m_im_size; j++){
                int dist_v = j - m_center_uv;
                // v position of a pixel of interested in map_scaled
                float px_v = v_c + dist_v * ratio_px;

                // if the pixel is out of range 6 sigma_n, fill 0
                if (dist_u * dist_u + dist_v * dist_v > m_r_sqr){
                    *map_list.mutable_data(num, 0, i, j) = 0;
                }else{
                    if (px_u >= 0 && px_v >= 0 && px_u < height && px_v < width){
                        *map_list.mutable_data(num, 0, i, j) = pick_pxl(map_scaled, px_u,
                                                                        px_v, m_interpolation);
                    }else{
                        // if the pixel if out of map_scaled, fill 31
                        *map_list.mutable_data(num, 0, i, j) = 31;
                    }
                }
            }
        }

    }

private:


    std::vector<Image> m_data;
    ssize_t m_im_size;
    array_f m_x_range, m_y_range, m_x_min, m_y_max, m_sigma_n;
    std::vector<ssize_t> m_shape_image;
    Interpolation m_interpolation;

    int m_r_sqr, m_center_uv;
    

    int32_t pick_pxl(Image map, float u_px, float v_px, Interpolation interpolation){
        int32_t px_value = 0;

        switch (interpolation) {
            case NEAREST:
                {int u = int(u_px + 0.5);
                int v = int(v_px + 0.5);
                px_value = *map.data(u, v);}
                break;
            default:
                std::cout << "Interpolation is not specified or invalid!" << std::endl;
                break;
        }
        return px_value;
    }

    
};


PYBIND11_PLUGIN(heightmap) {
    using Image = py::array_t<int32_t>;
    using array_f = std::vector<float>;
    py::module m("heightmap", "test docs");

    py::class_<HeightMap> heightmap(m, "HeightMap");

    heightmap.def(py::init<std::vector<Image>, ssize_t, array_f,
                  array_f, array_f, array_f, array_f>())
            .def(py::init<ssize_t>())
            .def("get_height_map", &HeightMap::get_height_map)
            .def("clip_scaled_map",
                static_cast<Image (HeightMap::*)(Image, float, float, float,float,
                                                float, float, float)>(&HeightMap::clip_scaled_map),
                "Clip Height map image by given parameters");

    py::enum_<HeightMap::Interpolation>(heightmap, "Interpolation")
        .value("NEAREST", HeightMap::Interpolation::NEAREST)
        .export_values();

    return m.ptr();
}
