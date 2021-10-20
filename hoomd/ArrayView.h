// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file ArrayView.h
    \brief Provides a Python list like view on an existing homogeneous buffer.
 */

#ifndef __ARRAY_VIEW_H__
#define __ARRAY_VIEW_H__

#ifndef __HIPCC__
#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#endif

// This class provides a Python list like interface to an arbitrary homogeneous data buffer.
//
// The intention is to expose arrays as list like arrays (this can already occur with std::vector
// thorough pybind11 automatically, however, in some cases such as statically sized arrays, the
// array_view is necessary). Currently this class only supports fixed size arrays or at least does
// not support changing size natively (perhaps the callback feature could enable this by changing
// buffer_size directly.
//
// The class provides an update functionality that can use a function like object that takes in a
// const array_view<value_type>* and returns void. This allow for callbacks to modify any
// other objects that may own the data being modified.
//
// The class currently does not support index, count, sort, reverse, or copy as these require type
// traits that involve much more logic. They can be added with some template metaprogramming though
// (the complexity of which is depenedent on how recent of a C++ standard we are willing to use -
// e.g. C++20 has concepts that make this fairly easy).
//
// NOTE:
//     Since the array_view template class is effectively a pointer to a data buffer that can
//     change after construction, IT IS NOT SAFE TO STORE POINTERS OR REFERENCES TO THIS IN PYTHON
//     OR C++ WHEN MODIFYING THE ORIGINAL DATA BUFFER which may change data location if size has
//     changed or otherwise invalidate the object's internal data pointer.  However, this class
//     should be safe to use if all references/pointers to it are acquired as late as possible and
//     destroyed as soon as possible.
//
//     A potential example of this is viewing the data of a ManagedArray with this class in Python.
//     If the array is resized (reallocated) while the array_view is still extant, then the pointer
//     stored in array_view will be invalid.
//
// WARNING:
//     In Python this class should only be used internally and never made user facing due to the
//     potential for SEGFAULTS and other data corruption fun.
//
// @param data_ pointer to the data
// @param buffer_size_ the maximum size of the buffer (this is similar to std::vector's capacity)
// @param size_ the current size of the used buffer (this is similar to std::vector's size)
// @param callback A potential callback function like object that takes in a pointer to the
//        array_view and returns void. The callback is called every time the data buffer is mutated
//        in any way. By using std::function, lambdas, function pointers, and classes like std::bind
//        can be used.
template<typename value_type> struct PYBIND11_EXPORT array_view
    {
    value_type* data;
    size_t buffer_size;
    size_t size;

    array_view(value_type* data_,
               const size_t buffer_size_,
               const size_t size_,
               const std::function<void(const array_view<value_type>*)> callback = nullptr)
        : data(data_), buffer_size(buffer_size_), size(size_), update_callback(callback)
        {
        }

    void insert(size_t index, value_type& value)
        {
        if (size == buffer_size)
            {
            throw std::runtime_error("Buffer is full.");
            }
        // Python appends to the list if insert > len(list).
        if (index > size)
            {
            index = size;
            }

        for (size_t i(size - 1); i >= index; --i)
            {
            data[i + 1] = data[i];
            if (i == 0)
                {
                break;
                }
            }
        data[index] = value;
        ++size;
        update();
        }

    void delItem(size_t index)
        {
        if (index >= buffer_size)
            {
            throw std::out_of_range("Index larger than buffer size.");
            }
        if (index >= size)
            {
            throw std::out_of_range("Cannot delete an item beyond the end of the list.");
            }
        for (size_t i = index; i < size - 1; ++i)
            {
            data[i] = data[i + 1];
            }
        --size;
        update();
        }

    value_type& getItem(size_t index) const
        {
        if (index >= buffer_size)
            {
            throw std::out_of_range("Index larger than buffer size.");
            }
        if (index >= size)
            {
            throw std::out_of_range("Cannot get an item beyond the end of the list.");
            }
        return data[index];
        }

    void setItem(size_t index, value_type& value)
        {
        if (index >= buffer_size)
            {
            throw std::out_of_range("Index larger than buffer size.");
            }
        if (index >= size)
            {
            throw std::out_of_range("Cannot set on an nonexistent index of the list.");
            }
        data[index] = value;
        update();
        }

    void append(value_type& value)
        {
        if (size == buffer_size)
            {
            throw std::runtime_error("Buffer is full.");
            }
        data[size] = value;
        ++size;
        update();
        }

    void extend(pybind11::object py_iterable)
        {
        auto list = py_iterable.cast<pybind11::list>();
        if (size + list.size() > buffer_size)
            {
            throw std::runtime_error("Buffer is full.");
            }
        for (auto& value : list)
            {
            data[size] = value.cast<value_type>();
            ++size;
            }
        update();
        }

    void clear()
        {
        size = 0;
        update();
        }

    value_type pop(size_t index)
        {
        if (index >= buffer_size)
            {
            throw std::out_of_range("Index larger than buffer size.");
            }
        if (index >= size)
            {
            throw std::out_of_range("Cannot pop an nonexistent index of the list.");
            }
        auto popped_value = data[index];
        delItem(index);
        update();
        return popped_value;
        }

    size_t len() const
        {
        return size;
        }

    private:
    const std::function<void(const array_view<value_type>*)> update_callback;

    // Call update_callback if provided. Allows for arbitrary logic to happen upon a buffer update.
    void update()
        {
        // check to see if callback function is null
        if (update_callback)
            {
            update_callback(this);
            }
        }
    };

// Helper function to make array_view objects
template<class value_type>
array_view<value_type> make_array_view(value_type* data,
                                       size_t buffer_size,
                                       size_t size,
                                       std::function<void(const array_view<value_type>*)> callback
                                       = nullptr)
    {
    return array_view<value_type>(data, buffer_size, size, callback);
    }

// Must manually specify all types exposed to Python in other classes. Notice that an init is not
// specified as these can only be exposed to Python but not created from Python.
template<class array_view_type> void export_array_view(pybind11::module m, const std::string& name)
    {
    pybind11::class_<array_view<array_view_type>>(m, name.c_str())
        .def("__len__", &array_view<array_view_type>::len)
        .def("insert", &array_view<array_view_type>::insert)
        .def("pop", &array_view<array_view_type>::pop)
        .def("clear", &array_view<array_view_type>::clear)
        .def("append", &array_view<array_view_type>::append)
        .def("extend", &array_view<array_view_type>::extend)
        .def("__delitem__", &array_view<array_view_type>::delItem)
        .def("__setitem__", &array_view<array_view_type>::setItem)
        .def("__getitem__", &array_view<array_view_type>::getItem);
    }
#endif
