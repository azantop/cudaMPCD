#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

#include "common/vector_3d.hpp"
#include "gpu_arrays.hpp"

namespace mpcd::cuda {
    /**
    * @brief A container that can be accessed by a 3D floating point vector returning elements of type T on a grid.
    *        Floating point x,y,z are rounded to the next ints.
    *        Internally this container wraps a std::vector< T >. Tecnically it is a hashmap using a vektor as key.
    */
    template< typename T >
    struct DeviceVolumeContainer
    {
        using Vector                 =       math::Vector;
        using Float              =       math::Float;
        using value_type             =       T;
        using reference              =       T&;
        using const_reference        = const T&;
        using iterator               =       T*;
        using const_iterator         = const T*;
        using reverse_iterator       =       std::reverse_iterator< iterator >;
        using const_reverse_iterator =       std::reverse_iterator< const_iterator >;
        using size_type              =       size_t;
        using difference_type        =       ptrdiff_t;

        private:

        Vector            edges,
                        inverse_edges,
                        shift;
        uint32_t          x_size,
                        y_size,
                        z_size,
                        xy_size;
        DeviceVector<T>   store;


        public:

        __host__ DeviceVolumeContainer(Vector const& size) : edges(size), inverse_edges(edges.getInverse()), shift(edges * Float(0.5)),
                                                            x_size(size.x), y_size(size.y), z_size(size.z), xy_size(size.x * size.y),
                                                            store(xy_size * z_size) {}

        DeviceVolumeContainer()                             = default;
        DeviceVolumeContainer(DeviceVolumeContainer const&) = default;
        DeviceVolumeContainer(DeviceVolumeContainer &&)     = default;

        DeviceVolumeContainer& operator = (DeviceVolumeContainer const&) = default;
        DeviceVolumeContainer& operator = (DeviceVolumeContainer &&)     = default;


        operator std::vector< value_type >& () { return store; }

        __host__            void      set(int i) { store.set( i ); }

        __host__ __device__ size_type size()      const { return store.size(); }
        __host__ __device__ size_type edge_x()    const { return x_size; }
        __host__ __device__ size_type edge_y()    const { return y_size; }
        __host__ __device__ size_type edge_z()    const { return z_size; }
        __host__ __device__ size_type face_xy()   const { return xy_size; }
        __host__ __device__ Vector    get_edges() const { return edges; }

        __host__ __device__ bool im_volumen(Vector const& position) const { return (round(position.scaledWith(inverse_edges)) == 0); }

        __device__ reference  operator[] (size_t const& idx)       { return store[ idx ]; }
        __device__ value_type operator[] (size_t const& idx) const { return store[ idx ]; }


        __device__ uint32_t get_index(Vector const& position) const {
            uint32_t x = static_cast<uint32_t>(floorf(position.x + shift.x)) % x_size;
            uint32_t y = static_cast<uint32_t>(floorf(position.y + shift.y)) % y_size;
            uint32_t z = static_cast<uint32_t>(floorf(position.z + shift.z)) % z_size;

            return x + (y * x_size) + (z * xy_size);
        }

        __device__ value_type&     operator[] (Vector const& position) {
            return store[get_index(position)];
        }

        __device__ value_type      operator[] ( Vector const& position ) const {
            return store[get_index(position)];
        }


        __device__ Vector get_position(uint32_t const& idx) const {
            return Vector({Float(0.5) + (idx % x_size),
                        Float(0.5) + ((idx % xy_size) / x_size),
                        Float(0.5) + (idx / xy_size)})
                    - shift;
        }

        __device__ Float get_z_idx(uint32_t const& idx) const{
            return idx / xy_size;
        }

        // data access:
        __device__ __host__ value_type*       data()       { return store.data(); }
        __device__ __host__ value_type const* data() const { return store.data(); }

        __device__ iterator       begin()        { return store.data(); }
        __device__ const_iterator begin()  const { return store.data(); }
        __device__ const_iterator cbegin() const { return store.data(); }
        __device__ iterator       end()          { return store.data() + store.size(); }
        __device__ const_iterator end()    const { return store.data() + store.size(); }
        __device__ const_iterator cend()   const { return store.data() + store.size(); }

        __device__ reverse_iterator       rbegin()
                { return reverse_iterator(store.data() + store.size()); }
        __device__ const_reverse_iterator rbegin()  const
                { return reverse_iterator(store.data() + store.size()); }
        __device__ reverse_iterator       rend()
                { return reverse_iterator(store.data()); }
        __device__ const_reverse_iterator rend()    const
                            { return reverse_iterator(store.data()); }
    };
} // namespace mpcd::cuda
