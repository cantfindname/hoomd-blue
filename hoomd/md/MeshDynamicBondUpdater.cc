// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file MeshDynamicBondUpdater.cc
    \brief Defines the MeshDynamicBondUpdater class
*/

#include "MeshDynamicBondUpdater.h"

#include <iostream>

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System definition
 *  \param rotational_diffusion The diffusion across time
 *  \param group the particles to diffusion rotation on
 */

MeshDynamicBondUpdater::MeshDynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                               std::shared_ptr<Integrator> integrator,
                                               std::shared_ptr<MeshDefinition> mesh)
    : Updater(sysdef), m_integrator(integrator), m_mesh(mesh)
    {
    assert(m_pdata);
    assert(m_integrator);
    assert(m_mesh);
    m_exec_conf->msg->notice(5) << "Constructing MeshDynamicBondUpdater" << endl;
    }

MeshDynamicBondUpdater::~MeshDynamicBondUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying MeshDynamicBondUpdater" << endl;
    }

/** Perform the needed calculations to update particle orientations
    \param timestep Current time step of the simulation
*/
void MeshDynamicBondUpdater::update(uint64_t timestep)
    {
    ArrayHandle<typename MeshBond::members_t> h_bonds(m_mesh->getMeshBondData()->getMembersArray(),
                                                      access_location::host,
                                                      access_mode::readwrite);
    ArrayHandle<typename MeshTriangle::members_t> h_triangles(
        m_mesh->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::readwrite);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // for each of the angles
    const unsigned int size = (unsigned int)m_mesh->getMeshBondData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        const typename MeshBond::members_t& bond = h_bonds.data[i];
        assert(bond.tag[0] < m_pdata->getMaximumTag() + 1);
        assert(bond.tag[1] < m_pdata->getMaximumTag() + 1);

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int tag_a = bond.tag[0];
        unsigned int idx_a = h_rtag.data[tag_a];
        unsigned int tag_b = bond.tag[1];
        unsigned int idx_b = h_rtag.data[tag_b];

        unsigned int tr_idx1 = bond.tag[2];
        unsigned int tr_idx2 = bond.tag[3];

        if (tr_idx1 == tr_idx2)
            continue;

        const typename MeshTriangle::members_t& triangle1 = h_triangles.data[tr_idx1];
        const typename MeshTriangle::members_t& triangle2 = h_triangles.data[tr_idx2];

        unsigned int tag_c = triangle1.tag[0];
        unsigned int idx_c = h_rtag.data[tag_c];

        unsigned int iterator = 1;
        while (idx_a == idx_c || idx_b == idx_c)
            {
            tag_c = triangle1.tag[iterator];
            idx_c = h_rtag.data[tag_c];
            iterator++;
            }

        unsigned int tag_d = triangle2.tag[0];
        unsigned int idx_d = h_rtag.data[tag_d];

        iterator = 1;
        while (idx_a == idx_d || idx_b == idx_d)
            {
            tag_d = triangle2.tag[iterator];
            idx_d = h_rtag.data[tag_d];
            iterator++;
            }

        Scalar energyDifference = 0;

        std::vector<std::shared_ptr<ForceCompute>> forces = m_integrator->getForces();

        for (auto& force : forces)
            {
            force->energyDiff(idx_a, idx_b, idx_c, idx_d);
            }

        if (energyDifference < 0)
            {
            typename MeshBond::members_t& bond_n = h_bonds.data[i];
            typename MeshTriangle::members_t& triangle1_n = h_triangles.data[tr_idx1];
            typename MeshTriangle::members_t& triangle2_n = h_triangles.data[tr_idx2];

            triangle1_n.tag[0] = tag_a;
            triangle1_n.tag[1] = tag_c;
            triangle1_n.tag[2] = tag_d;
            triangle1_n.tag[3] = i;

            triangle2_n.tag[0] = tag_b;
            triangle2_n.tag[1] = tag_d;
            triangle2_n.tag[2] = tag_c;
            triangle2_n.tag[3] = i;

            bond_n.tag[0] = tag_c;
            bond_n.tag[1] = tag_d;

            h_bonds.data[i] = bond_n;

            for (int j = 3; j < 6; j++)
                {
                int k = triangle1.tag[j];
                if (k == i)
                    {
                    bond_n = h_bonds.data[k];

                    unsigned int tr_idx;
                    if (bond_n.tag[0] == tag_a || bond_n.tag[1] == tag_a)
                        {
                        tr_idx = tr_idx1;
                        triangle1_n.tag[4] = k;
                        }
                    else
                        {
                        tr_idx = tr_idx2;
                        triangle2_n.tag[4] = k;
                        }

                    if (bond_n.tag[2] == tr_idx1 || bond_n.tag[2] == tr_idx2)
                        bond_n.tag[2] = tr_idx;
                    else
                        bond_n.tag[3] = tr_idx;
                    h_bonds.data[k] = bond_n;
                    }

                k = triangle2.tag[j];
                if (k == i)
                    {
                    bond_n = h_bonds.data[k];

                    unsigned int tr_idx;
                    if (bond_n.tag[0] == tag_a || bond_n.tag[1] == tag_a)
                        {
                        tr_idx = tr_idx1;
                        triangle1_n.tag[5] = k;
                        }
                    else
                        {
                        tr_idx = tr_idx2;
                        triangle2_n.tag[5] = k;
                        }

                    if (bond_n.tag[2] == tr_idx1 || bond_n.tag[2] == tr_idx2)
                        bond_n.tag[2] = tr_idx;
                    else
                        bond_n.tag[3] = tr_idx;
                    h_bonds.data[k] = bond_n;
                    }
                }

            h_triangles.data[tr_idx1] = triangle1_n;
            h_triangles.data[tr_idx2] = triangle2_n;
            }
        }
    }

namespace detail
    {
void export_MeshDynamicBondUpdater(pybind11::module& m)
    {
    pybind11::class_<MeshDynamicBondUpdater, Updater, std::shared_ptr<MeshDynamicBondUpdater>>(
        m,
        "MeshDynamicBondUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Integrator>,
                            std::shared_ptr<MeshDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
