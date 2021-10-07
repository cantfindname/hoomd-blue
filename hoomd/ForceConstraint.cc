// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

#include "ForceConstraint.h"

namespace py = pybind11;

using namespace std;

/*! \file ForceConstraint.cc
    \brief Contains code for the ForceConstraint class
*/

namespace hoomd
    {
/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
 */
ForceConstraint::ForceConstraint(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef) { }

/*! Does nothing in the base class
    \param timestep Current timestep
*/
void ForceConstraint::computeForces(uint64_t timestep) { }

namespace detail
    {
void export_ForceConstraint(py::module& m)
    {
    py::class_<ForceConstraint, ForceCompute, std::shared_ptr<ForceConstraint>>(m,
                                                                                "ForceConstraint")
        .def(py::init<std::shared_ptr<SystemDefinition>>());
    }
    } // end namespace detail

    } // end namespace hoomd
