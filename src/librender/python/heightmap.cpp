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
    }

    auto get_height_map(py::array_t<float> in_pos, array_i mesh_id){

        clock_t start = clock();
        std::cout << "get_map_start" << std::endl;

        ssize_t n_sample = mesh_id.size();

        std::vector<array_i> result(n_sample);

        std::cout << "shape: " << result.size() << ", " << m_shape_result[0] << ", "
                  << m_shape_result[1] << ", " << m_shape_result[2] << std::endl;

        for(int i = 0; i < n_sample; i++){
            for(int j = 0; j < m_im_size; j++){
                for(int k = 0; k < m_im_size; k++){

                }
            }
        }

        return result;
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
