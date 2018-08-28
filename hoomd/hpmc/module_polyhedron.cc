// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "IntegratorHPMCMonoImplicitNew.h"
#include "ComputeFreeVolume.h"

#include "ShapePolyhedron.h"
#include "AnalyzerSDF.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"
#include "ExternalCallback.h"

#include "UpdaterExternalFieldWall.h"
#include "UpdaterRemoveDrift.h"
#include "UpdaterMuVT.h"
#include "UpdaterMuVTImplicit.h"
#include "UpdaterClusters.h"
#include "UpdaterClustersImplicit.h"

#ifdef ENABLE_CUDA
#include "IntegratorHPMCMonoGPU.h"
#include "IntegratorHPMCMonoImplicitGPU.h"
#include "IntegratorHPMCMonoImplicitNewGPU.h"
#include "ComputeFreeVolumeGPU.h"
#endif

namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the base HPMCMono integrators
void export_polyhedron(py::module& m)
    {
    export_IntegratorHPMCMono< ShapePolyhedron >(m, "IntegratorHPMCMonoPolyhedron");
    #ifdef ENABLE_HPMC_REINSERT
    export_IntegratorHPMCMonoImplicit< ShapePolyhedron >(m, "IntegratorHPMCMonoImplicitPolyhedron");
    #endif
    export_IntegratorHPMCMonoImplicitNew< ShapePolyhedron >(m, "IntegratorHPMCMonoImplicitNewPolyhedron");
    export_ComputeFreeVolume< ShapePolyhedron >(m, "ComputeFreeVolumePolyhedron");
    // export_AnalyzerSDF< ShapePolyhedron >(m, "AnalyzerSDFPolyhedron");
    export_UpdaterMuVT< ShapePolyhedron >(m, "UpdaterMuVTPolyhedron");
    export_UpdaterClusters< ShapePolyhedron >(m, "UpdaterClustersPolyhedron");
    #ifdef ENABLE_HPMC_REINSERT
    export_UpdaterClustersImplicit< ShapePolyhedron, IntegratorHPMCMonoImplicit<ShapePolyhedron> >(m, "UpdaterClustersImplicitPolyhedron");
    #endif
    export_UpdaterClustersImplicit< ShapePolyhedron, IntegratorHPMCMonoImplicitNew<ShapePolyhedron> >(m, "UpdaterClustersImplicitNewPolyhedron");
    #ifdef ENABLE_HPMC_REINSERT
    export_UpdaterMuVTImplicit< ShapePolyhedron, IntegratorHPMCMonoImplicit<ShapePolyhedron> >(m, "UpdaterMuVTImplicitPolyhedron");
    #endif
    export_UpdaterMuVTImplicit< ShapePolyhedron, IntegratorHPMCMonoImplicitNew<ShapePolyhedron> >(m, "UpdaterMuVTImplicitNewPolyhedron");

    export_ExternalFieldInterface<ShapePolyhedron>(m, "ExternalFieldPolyhedron");
    export_LatticeField<ShapePolyhedron>(m, "ExternalFieldLatticePolyhedron");
    export_ExternalFieldComposite<ShapePolyhedron>(m, "ExternalFieldCompositePolyhedron");
    export_RemoveDriftUpdater<ShapePolyhedron>(m, "RemoveDriftUpdaterPolyhedron");
    export_ExternalFieldWall<ShapePolyhedron>(m, "WallPolyhedron");
    export_UpdaterExternalFieldWall<ShapePolyhedron>(m, "UpdaterExternalFieldWallPolyhedron");
    export_ExternalCallback<ShapePolyhedron>(m, "ExternalCallbackPolyhedron");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapePolyhedron >(m, "IntegratorHPMCMonoGPUPolyhedron");
    #ifdef ENABLE_HPMC_REINSERT
    export_IntegratorHPMCMonoImplicitGPU< ShapePolyhedron >(m, "IntegratorHPMCMonoImplicitGPUPolyhedron");
    #endif
    export_IntegratorHPMCMonoImplicitNewGPU< ShapePolyhedron >(m, "IntegratorHPMCMonoImplicitNewGPUPolyhedron");
    export_ComputeFreeVolumeGPU< ShapePolyhedron >(m, "ComputeFreeVolumeGPUPolyhedron");
    #endif
    }

}
