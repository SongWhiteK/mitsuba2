#include <iostream>
#include <mitsuba/python/python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <time.h>

template<typename T>
class HeightMap {
private:
    py::array_t<T> m_data;
    ssize_t m_im_size;

public:
    HeightMap(py::array_t<T> map_list, ssize_t im_size){
        m_data = map_list;
        m_im_size = im_size;
    }

    auto print_size(){
        const auto &buff_info = m_data.request();
        const auto &shape = buff_info.shape;
        const auto &ndim = buff_info.ndim;

        py::array_t<T> y{shape};


        return y;
    }

    auto get_height_map(py::array_t<T> in_pos, py::array_t<T> mesh_id){
        clock_t start = clock();
        std::cout << "get_map_start" << std::endl;
        const auto &buff_id = mesh_id.request();
        ssize_t n_sample = buff_id.shape[0];

        std::vector<ssize_t> shape_result{n_sample, m_im_size, m_im_size};

        py::array_t<T> result{shape_result};

        return result;
    }

    
};


PYBIND11_PLUGIN(heightmap) {
    py::module m("heightmap", "test docs");

    py::class_<HeightMap<int32_t>>(m, "HeightMap")
        .def(py::init<py::array_t<int32_t>, ssize_t>())
        .def("print_size", &HeightMap<int32_t>::print_size)
        .def("get_height_map", &HeightMap<int32_t>::get_height_map);

    return m.ptr();
}
