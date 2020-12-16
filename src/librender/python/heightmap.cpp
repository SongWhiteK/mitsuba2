#include <iostream>
#include <mitsuba/python/python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <time.h>

class HeightMap {
    using array_i = py::array_t<int32_t>;
    using array_f = std::vector<float>;
private:
    std::vector<array_i> m_data;
    ssize_t m_im_size;
    array_f m_x_range, m_y_range, m_x_min, m_y_max, m_sigma_n;
    std::vector<ssize_t> m_shape_result;


public:
    HeightMap(std::vector<array_i> map_list, ssize_t im_size, array_f x_range,
              array_f y_range, array_f sigma_n, array_f x_min, array_f y_max){
        m_data = map_list;
        m_im_size = im_size;
        m_x_range = x_range;
        m_y_range = y_range;
        m_sigma_n = sigma_n;
        m_x_min = x_min;
        m_y_max = y_max;
        m_shape_result = std::vector<ssize_t>{1, im_size, im_size};
    }

    auto get_height_map(py::array_t<float> in_pos, array_i mesh_id){

        clock_t start = clock();
        std::cout << "get_map_start" << std::endl;

        ssize_t n_sample = mesh_id.size();

        std::vector<array_i> result(n_sample);

        std::cout << "shape: " << result.size() << ", " << m_shape_result[0] << ", "
                  << m_shape_result[1] << ", " << m_shape_result[2] << std::endl;

        // Roop of sampling for each ray
        for (int i = 0; i < n_sample; i++){
            int32_t id_i = *mesh_id.data(i);
            if(id_i == 0) continue;

            result[i] = clip_map(i, i+1);
        }

        return result;
    }

    array_i clip_map(int map_id, int32_t num){
        array_i cliped_map{m_shape_result};

        for (int i = 0; i < m_im_size; i++){
            for(int j = 0; j < m_im_size; j++){
                *cliped_map.mutable_data(0, i, j) = num;
            }
        }

        return cliped_map;
    }

    
};


PYBIND11_PLUGIN(heightmap) {
    using array_i = py::array_t<int32_t>;
    using array_f = std::vector<float>;
    py::module m("heightmap", "test docs");

    py::class_<HeightMap>(m, "HeightMap")
        .def(py::init<std::vector<array_i>, ssize_t, array_f,
             array_f, array_f, array_f, array_f>())
        .def("get_height_map", &HeightMap::get_height_map);

    return m.ptr();
}
