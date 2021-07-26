// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// **********************
// This is a simple example code written for no function purpose other then to demonstrate the steps
// needed to write a c++ source code plugin for HOOMD-Blue. This example includes an example Updater
// class, but it can just as easily be replaced with a ForceCompute, Integrator, or any other C++
// code at all.

// inclusion guard
#ifndef _REMOVE_DRIFT_UPDATER_H_
#define _REMOVE_DRIFT_UPDATER_H_

/*! \file ExampleUpdater.h
    \brief Declaration of ExampleUpdater
*/

// First, hoomd.h should be included

#include "IntegratorHPMCMono.h"
#include "hoomd/Updater.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hpmc
    {
// (if you really don't want to include the whole hoomd.h, you can include individual files IF AND
// ONLY IF hoomd_config.h is included first) For example:
//
// #include "hoomd/Updater.h"

// Second, we need to declare the class. One could just as easily use any class in HOOMD as a
// template here, there are no restrictions on what a template can do

//! A nonsense particle updater written to demonstrate how to write a plugin
/*! This updater simply sets all of the particle's velocities to 0 when update() is called.
 */
template<class Shape> class RemoveDriftUpdater : public Updater
    {
    public:
    //! Constructor
    RemoveDriftUpdater(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                       pybind11::array_t<double> ref_positions)
        : Updater(sysdef), m_mc(mc)
        {
        setReferencePositions(ref_positions);
        }

    //! Set reference positions from a Nx3 numpy array
    void setReferencePositions(const pybind11::array_t<double> ref_pos)
        {
        const size_t N = ref_pos.request().shape[0];
        const size_t dim = ref_pos.request().shape[1];
        if (N != this->m_pdata->getNGlobal() || dim != 3)
            {
            throw std::runtime_error("The array must be of shape (N_particles, 3).");
            }
        const double* rawdata = static_cast<double*>(ref_pos.request().ptr);
        m_ref_positions.resize(m_pdata->getNGlobal());
        for (size_t i = 0; i < N; i++)
            {
            const size_t array_index = i * 3;
            this->m_ref_positions[i] = vec3<Scalar>(rawdata[array_index],
                                                    rawdata[array_index + 1],
                                                    rawdata[array_index + 2]);
            }

#ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            bcast(m_ref_positions, 0, m_exec_conf->getMPICommunicator());
            }
#endif
        }

    //! Get reference positions as a Nx3 numpy array
    pybind11::array_t<double> getReferencePositions() const
        {
        std::vector<size_t> dims(2);
        dims[0] = this->m_ref_positions.size();
        dims[1] = 3;
        // the cast from vec3<Scalar>* to Scalar* is safe since vec3 is tightly packed without any
        // padding. This also makes a copy so, modifications of this array do not effect the
        // original reference positions.
        const auto reference_array = pybind11::array_t<double>(
            dims,
            reinterpret_cast<const double*>(this->m_ref_positions.data()));
        // This is necessary to expose the array in a read only fashion through C++
        reinterpret_cast<pybind11::detail::PyArray_Proxy*>(reference_array.ptr())->flags
            &= ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        return reference_array;
        }

    //! Take one timestep forward
    virtual void update(uint64_t timestep)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);
        ArrayHandle<int3> h_image(this->m_pdata->getImages(),
                                  access_location::host,
                                  access_mode::readwrite);
        const BoxDim& box = this->m_pdata->getGlobalBox();
        const vec3<Scalar> origin(this->m_pdata->getOrigin());
        vec3<Scalar> rshift;
        rshift.x = rshift.y = rshift.z = 0.0f;

        for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
            {
            unsigned int tag_i = h_tag.data[i];
            // read in the current position and orientation
            vec3<Scalar> postype_i = vec3<Scalar>(h_postype.data[i]) - origin;
            int3 tmp_image = make_int3(0, 0, 0);
            box.wrap(postype_i, tmp_image);
            const vec3<Scalar> dr = postype_i - m_ref_positions[tag_i];
            rshift += vec3<Scalar>(box.minImage(vec_to_scalar3(dr)));
            }

#ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            Scalar r[3] = {rshift.x, rshift.y, rshift.z};
            MPI_Allreduce(MPI_IN_PLACE,
                          &r[0],
                          3,
                          MPI_HOOMD_SCALAR,
                          MPI_SUM,
                          m_exec_conf->getMPICommunicator());
            rshift.x = r[0];
            rshift.y = r[1];
            rshift.z = r[2];
            }
#endif

        rshift /= Scalar(this->m_pdata->getNGlobal());

        for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
            {
            // read in the current position and orientation
            Scalar4 postype_i = h_postype.data[i];
            const vec3<Scalar> r_i = vec3<Scalar>(postype_i);
            h_postype.data[i] = vec_to_scalar4(r_i - rshift, postype_i.w);
            box.wrap(h_postype.data[i], h_image.data[i]);
            }

        m_mc->invalidateAABBTree();
        // migrate and exchange particles
        m_mc->communicate(true);
        }

    protected:
    std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc;
    std::vector<vec3<Scalar>> m_ref_positions;
    };

//! Export the ExampleUpdater class to python
template<class Shape> void export_RemoveDriftUpdater(pybind11::module& m, std::string name)
    {
    using pybind11::class_;
    pybind11::class_<RemoveDriftUpdater<Shape>,
                     Updater,
                     std::shared_ptr<RemoveDriftUpdater<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            pybind11::array_t<Scalar>>())
        .def_property("reference_positions",
                      &RemoveDriftUpdater<Shape>::getReferencePositions,
                      &RemoveDriftUpdater<Shape>::setReferencePositions);
    }
    } // namespace hpmc

#endif // _REMOVE_DRIFT_UPDATER_H_
