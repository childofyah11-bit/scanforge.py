import numpy as np
import trimesh

import scanforge


def test_repair_mesh_returns_watertight_cube():
    cube = trimesh.creation.box()
    repaired = scanforge.repair_mesh(cube)

    assert repaired.is_watertight
    assert len(repaired.faces) > 0
    assert repaired.volume > 0


def test_scale_mesh_scales_uniformly():
    cube = trimesh.creation.box()
    scaled = scanforge.scale_mesh(cube, 2.0)

    assert np.allclose(scaled.extents, cube.extents * 2.0)
    assert np.isclose(scaled.volume, cube.volume * 8.0)


def test_make_printable_keeps_cube_printable():
    cube = trimesh.creation.box()
    printable = scanforge.make_printable(cube)

    assert printable.is_watertight
    assert len(printable.faces) > 0
    assert printable.volume > 0
    assert np.isclose(printable.bounds[0][2], 0.0)
