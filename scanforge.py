#!/usr/bin/env python3
"""ScanForge - a simple object-to-printable STL pipeline for Bambu Studio.

One tool, the whole chain:

  CAPTURE      photo-capture checklist for best scans
  RECONSTRUCT  photos -> 3D mesh
                engine auto-detect (best available wins):
                  1. COLMAP      (pro, if installed)
                  2. Meshroom    (pro, if installed)
                  3. BUILT-IN    experimental SfM engine (OpenCV, beta quality)
  REPAIR       de-sliver, dedupe, fix normals, fill holes, remove noise blobs
  SCALE        set true size in millimeters
  PREP         place on the bed and export a slicer-friendly STL
  EXPORT       print-ready binary STL + honest watertight verdict
  HOLLOW       shell the solid to a wall thickness with drain holes

This flow is designed around common Bambu Studio expectations: watertight meshes,
zero-based Z placement, millimeter sizing, and simple, slicer-friendly STL output.

Install core:   pip install trimesh numpy
Reconstruction: pip install opencv-contrib-python   (built-in beta engine)
Optional pro:   pip install open3d   (nicer meshing) and/or install COLMAP/Meshroom
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
    import trimesh
except ImportError:
    print("Missing libraries. Install them with:  pip install trimesh numpy")
    sys.exit(1)

PHOTO_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SUPPORTED_EXPORTS = {"stl", "obj", "ply", "glb", "off"}

CAPTURE_STEPS = [
    "1. Pick an object with surface detail and NO mirror/glass parts.",
    "2. Even light - overcast window light beats one harsh lamp.",
    "3. Plain low-contrast background (big white poster board).",
    "4. Walk a full circle at eye level, a photo every 15-20 degrees.",
    "5. Repeat from 45 degrees down, then a few from above.",
    "6. Same distance all the way around, tap focus once, never zoom.",
    "7. Lock exposure if your camera app allows it.",
    "8. 40-80 sharp photos. Blur is the #1 scan killer.",
]


def cmd_guide(args):
    print("\n".join(CAPTURE_STEPS))
    print("\nThen run:  python scanforge.py scan ./photos --size 80")
    print("ScanForge picks the best engine you have and does the rest.")


def find_engine():
    """Return the best available reconstruction engine and callable."""
    if shutil.which("colmap"):
        return "COLMAP", run_colmap
    if shutil.which("meshroom_batch") or shutil.which("meshroom_photogrammetry"):
        return "Meshroom", run_meshroom
    try:
        import cv2  # noqa: F401
        return "ScanForge-Beta (OpenCV SfM)", run_builtin_sfm
    except ImportError:
        return None, None


def run_colmap(photo_dir, out_dir):
    """Run a basic COLMAP pipeline if the executable is installed."""
    db = out_dir / "colmap.db"
    sparse = out_dir / "sparse"
    dense = out_dir / "dense"
    for d in (sparse, dense):
        d.mkdir(parents=True, exist_ok=True)
    cmds = [
        ["colmap", "feature_extractor", "--database_path", str(db), "--image_path", str(photo_dir)],
        ["colmap", "exhaustive_matcher", "--database_path", str(db)],
        ["colmap", "mapper", "--database_path", str(db), "--image_path", str(photo_dir), "--output_path", str(sparse)],
        ["colmap", "image_undistorter", "--image_path", str(photo_dir), "--input_path", str(sparse / "0"), "--output_path", str(dense)],
        ["colmap", "patch_match_stereo", "--workspace_path", str(dense)],
        ["colmap", "stereo_fusion", "--workspace_path", str(dense), "--output_path", str(out_dir / "fused.ply")],
    ]
    for c in cmds:
        print("  $", " ".join(str(x) for x in c))
        subprocess.run(c, check=False)
    return out_dir / "fused.ply"


def run_meshroom(photo_dir, out_dir):
    exe = shutil.which("meshroom_batch") or shutil.which("meshroom_photogrammetry")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not exe:
        return None
    subprocess.run([exe, "--input", str(photo_dir), "--output", str(out_dir)], check=False)
    model = out_dir / "texturedMesh.obj"
    return model if model.exists() else None


def run_builtin_sfm(photo_dir, out_dir):
    """A lightweight OpenCV fallback for small, textured objects."""
    try:
        import cv2
    except ImportError:
        print("Built-in engine needs OpenCV:  pip install opencv-contrib-python")
        return None

    try:
        import open3d as o3d
        have_o3d = True
    except ImportError:
        have_o3d = False

    photos = sorted(p for p in Path(photo_dir).iterdir() if p.suffix.lower() in PHOTO_EXT)
    if len(photos) < 3:
        print("Need at least 3 photos.")
        return None

    print(f"  [beta] loading {len(photos)} photos...")
    imgs, kps, dess = [], [], []
    target_w = 1200
    for p in photos:
        img = cv2.imread(str(p))
        if img is None:
            continue
        scale = target_w / img.shape[1]
        img = cv2.resize(img, (target_w, int(img.shape[0] * scale)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=4000)
        kp, desc = sift.detectAndCompute(gray, None)
        if kp is None or len(kp) < 500:
            continue
        imgs.append(img)
        kps.append(kp)
        dess.append(desc)

    if len(imgs) < 3:
        print("  Not enough usable photos (blur or low texture).")
        return None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))

    def match_pair(i, j):
        matches = flann.knnMatch(dess[i], dess[j], k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return good

    best = (0, 1, [])
    for j in range(1, min(len(imgs), 8)):
        g = match_pair(0, j)
        if len(g) > len(best[2]):
            best = (0, j, g)
    i0, i1, good = best
    if len(good) < 100:
        print("  Photos do not overlap enough. Re-shoot with more overlap.")
        return None

    focal = float(max(imgs[0].shape[0], imgs[0].shape[1]))
    cx, cy = imgs[0].shape[1] / 2, imgs[0].shape[0] / 2
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=float)
    pts0 = np.float32([kps[i0][m.queryIdx].pt for m in good])
    pts1 = np.float32([kps[i1][m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask is None:
        print("  Essential matrix estimation failed for the anchor pair.")
        return None

    _, R, t, _ = cv2.recoverPose(E, pts0, pts1, K, mask=mask.copy())
    if R is None or t is None:
        print("  Camera pose recovery failed for the anchor pair.")
        return None

    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t])
    q4 = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
    cloud = (q4[:3] / q4[3]).T
    norms = np.linalg.norm(cloud, axis=1)
    if norms.size == 0:
        print("  No valid 3D points were triangulated from the anchor pair.")
        return None

    valid = (cloud[:, 2] > 0) & (norms < np.median(norms) * 5)
    cloud = cloud[valid]
    colors = np.array([imgs[i0][int(y), int(x)][::-1] for x, y in pts0[valid]], dtype=np.float32) / 255.0

    print(f"  [beta] anchor pair photos {i0}-{i1}: {len(cloud)} seed points")
    print("  [beta] adding remaining photos (pose tracking)...")
    for j in range(len(imgs)):
        if j in (i0, i1):
            continue
        good_j = match_pair(i1, j)
        if len(good_j) < 80:
            continue
        pj = np.float32([kps[i1][m.queryIdx].pt for m in good_j])
        ok, rvec, tvec, _ = cv2.solvePnPRansac(cloud[: len(pj)], pj, K, None)
        if not ok:
            continue
        R_j = cv2.Rodrigues(rvec)[0]
        P_j = K @ np.hstack([R_j, tvec])

        m01 = match_pair(i0, j)
        if len(m01) < 80:
            continue
        a = np.float32([kps[i0][m.queryIdx].pt for m in m01])
        b = np.float32([kps[j][m.trainIdx].pt for m in m01])
        q4 = cv2.triangulatePoints(P0, P_j, a.T, b.T)
        q3 = (q4[:3] / q4[3]).T
        norms_q = np.linalg.norm(q3, axis=1)
        if norms_q.size == 0:
            continue
        ok_q = (q3[:, 2] > 0) & (norms_q < np.median(norms_q) * 5 + 1e-9)
        q3 = q3[ok_q]
        cols = np.array([imgs[j][int(y), int(x)][::-1] for x, y in b[ok_q]], dtype=np.float32) / 255.0
        cloud = np.vstack([cloud, q3])
        colors = np.vstack([colors, cols])

    print(f"  [beta] point cloud: {len(cloud)} points")
    out_dir.mkdir(parents=True, exist_ok=True)
    if have_o3d:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=float(np.ptp(cloud, axis=0).max()) / 200)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
        pcd.orient_normals_consistent_tangent_plane(30)
        mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        mesh_o3d = mesh_o3d.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=pcd.get_min_bound() - 1e-6,
                max_bound=pcd.get_max_bound() + 1e-6,
            )
        )
        out_path = out_dir / "scanforge_mesh.obj"
        o3d.io.write_triangle_mesh(str(out_path), mesh_o3d)
        print(f"  [beta] Poisson mesh -> {out_path}")
        return out_path

    ply = out_dir / "scanforge_cloud.ply"
    header = (
        f"ply\nformat ascii 1.0\nelement vertex {len(cloud)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float red\nproperty float green\nproperty float blue\nend_header\n"
    )
    data = "\n".join(
        f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}"
        for p, c in zip(cloud, colors)
    )
    ply.write_text(header + data + "\n")
    print(f"  [beta] no Open3D - saved raw point cloud -> {ply}")
    print("  Install open3d for automatic meshing:  pip install open3d")
    return ply


def cmd_scan(args):
    photo_dir = Path(args.dir)
    photos = [p for p in photo_dir.iterdir() if p.suffix.lower() in PHOTO_EXT]
    if len(photos) < 10:
        print(f"Only {len(photos)} photos in {photo_dir} - you want 40+.")
        return
    print(f"Found {len(photos)} photos.")
    name, engine = find_engine()
    if name is None:
        print("No engine available. Either install opencv-contrib-python for the")
        print("built-in beta engine, or COLMAP/Meshroom for pro quality.")
        return
    print(f"Engine: {name}")
    out_dir = photo_dir / "scanforge_out"
    out_dir.mkdir(exist_ok=True)
    model = engine(photo_dir, out_dir)
    if not model:
        print("Reconstruction failed - see messages above.")
        return
    print(f"\nMesh ready: {model}")
    if args.size:
        print("Now making it printable...\n")
        args_make = argparse.Namespace(model=str(model), size=args.size, out=args.out)
        cmd_make_printable(args_make)


def _safe_scene_to_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            raise ValueError("Scene has no geometry.")
        mesh = scene_or_mesh.dump(concatenate=True)
        if mesh is None:
            raise ValueError("Scene dump produced no mesh.")
        return mesh
    return scene_or_mesh


def validate_mesh(mesh, *, context="Mesh"):
    """Reject empty or degenerate mesh inputs early with a useful message."""
    mesh = _safe_scene_to_mesh(mesh)
    if mesh is None:
        raise ValueError(f"{context} is empty.")
    if not hasattr(mesh, "vertices") or len(getattr(mesh, "vertices", [])) == 0:
        raise ValueError(f"{context} has no vertices.")
    if not hasattr(mesh, "faces") or len(getattr(mesh, "faces", [])) == 0:
        raise ValueError(f"{context} has no faces.")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"{context} contains NaN or infinite values.")
    if float(np.max(mesh.extents)) <= 0.0:
        raise ValueError(f"{context} is degenerate and has zero or negative scale.")
    return mesh


def load_mesh(path):
    mesh_path = Path(path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"File not found: {mesh_path}")
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False, silent=True)
    if mesh is None:
        raise ValueError(f"Could not read a mesh from {mesh_path}")
    mesh = _safe_scene_to_mesh(mesh)
    return validate_mesh(mesh, context=f"Mesh file '{mesh_path}'")


def save_mesh(mesh, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower().lstrip(".")
    if ext not in SUPPORTED_EXPORTS:
        raise ValueError(f"Unsupported output format '{ext}'. Use one of: {sorted(SUPPORTED_EXPORTS)}")
    payload = trimesh.exchange.export.export_mesh(mesh, file_type=ext)
    if payload is None:
        raise ValueError(f"Mesh export failed for {out}")
    out.write_bytes(payload)
    return out


def mesh_report(mesh):
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding": bool(mesh.is_winding_consistent),
        "size_mm": tuple(round(float(e), 1) for e in mesh.extents),
    }


def print_report(rep, title="Mesh report"):
    print(f"--- {title} ---")
    for key, value in rep.items():
        print(f" {key + '.':14}{value}")


def _largest_component(mesh, *, prefer_watertight=False):
    """Return the largest useful component from a mesh."""
    if not hasattr(mesh, "split"):
        return mesh

    try:
        parts = mesh.split(only_watertight=False)
    except Exception:
        return mesh

    if not parts:
        return mesh

    def score(part):
        area = max(1, len(part.faces))
        if prefer_watertight:
            area += 100000 if getattr(part, "is_watertight", False) else 0
        return area

    return max(parts, key=score)


def repair_mesh(mesh):
    """Repair a mesh by removing bad elements and keeping the largest valid volume."""
    mesh = validate_mesh(mesh, context="Mesh to repair")
    repaired = mesh.copy()

    for method in ("remove_duplicate_faces", "remove_unreferenced_vertices", "merge_vertices"):
        try:
            getattr(repaired, method)()
        except Exception:
            pass

    for method in ("update_faces", "fix_normals"):
        try:
            if method == "update_faces":
                repaired.update_faces(repaired.nondegenerate_faces())
                repaired.update_faces(repaired.unique_faces())
            else:
                repaired.fix_normals()
        except Exception:
            pass

    try:
        repaired.process(validate=True)
    except Exception:
        pass

    try:
        repaired.fill_holes()
    except Exception:
        pass

    try:
        repaired.remove_unreferenced_vertices()
    except Exception:
        pass

    if len(repaired.faces) == 0:
        raise ValueError("Mesh has no faces to repair")

    repaired = _largest_component(repaired, prefer_watertight=True)
    repaired = validate_mesh(repaired, context="Repaired mesh")
    return repaired


def scale_mesh(mesh, factor):
    """Scale a mesh uniformly around its centroid and keep the output valid."""
    mesh = validate_mesh(mesh, context="Mesh to scale")
    if factor <= 0:
        raise ValueError(f"Scale factor must be positive, got {factor!r}")
    scaled = mesh.copy()
    scaled.apply_scale(factor)
    return validate_mesh(scaled, context="Scaled mesh")


def upgrade_mesh(mesh, *, size=None):
    """Repair, optionally scale, and place a mesh flat on the bed."""
    repaired = repair_mesh(mesh)

    if size is not None:
        size = float(size)
        if size <= 0:
            raise ValueError(f"size must be positive, got {size!r}")
        max_extent = float(max(repaired.extents))
        if max_extent <= 0:
            raise ValueError("Mesh extents are invalid; cannot scale to a print size.")
        repaired = scale_mesh(repaired, size / max_extent)

    bounds = repaired.bounds
    min_x, min_y, min_z = bounds[0]
    repaired.apply_translation([-min_x, -min_y, -min_z])

    bounds = repaired.bounds
    if np.isfinite(bounds).all():
        z_floor = float(bounds[0][2])
        if z_floor < 0.0:
            repaired.apply_translation([0.0, 0.0, -z_floor])

    return validate_mesh(repaired, context="Print-ready mesh")


def make_printable(mesh, *, size=None):
    """Repair a mesh, optionally resize it, and place it on the bed in Bambu Studio-friendly coordinates."""
    return upgrade_mesh(mesh, size=size)


def find_lowest_points(mesh, count=2):
    """Return the lowest vertices by Z coordinate."""
    if count <= 0:
        return []
    order = np.argsort(mesh.vertices[:, 2])
    return [int(idx) for idx in order[: max(1, min(count, len(order)))] ]


def cmd_check(args):
    mesh = load_mesh(args.model)
    print_report(mesh_report(mesh), "Mesh report")


def cmd_repair(args):
    mesh = load_mesh(args.model)
    before = mesh_report(mesh)
    repaired = repair_mesh(mesh)
    after = mesh_report(repaired)
    print_report(before, "Before")
    print()
    print_report(after, "After repair")
    out = Path(args.out) if args.out else Path(args.model).with_name(Path(args.model).stem + "_repaired.stl")
    save_mesh(repaired, out)
    print(f"\nSaved -> {out}")
    if not after["watertight"]:
        print("Heads up: still not watertight - big holes need a re-scan.")


def cmd_scale(args):
    mesh = load_mesh(args.model)
    out = Path(args.out) if args.out else None
    scaled = scale_mesh(mesh, args.size / float(max(mesh.extents)))
    print(f"Scaled x{args.size / float(max(mesh.extents)):.4f}: longest side = {args.size} mm")
    if out:
        save_mesh(scaled, out)
        print(f"Saved -> {out}")


def cmd_hollow(args):
    """Hollow a mesh to a wall thickness and punch simple drain holes at the lowest points."""
    mesh = load_mesh(args.model)
    mesh = repair_mesh(mesh)

    if args.wall <= 0:
        raise ValueError(f"wall thickness must be positive, got {args.wall!r}")
    if args.drain_holes <= 0:
        raise ValueError(f"drain_holes must be positive, got {args.drain_holes!r}")

    try:
        from scipy import ndimage
    except ImportError:
        print("Hollowing needs scipy:  pip install scipy")
        return

    pitch = float(max(mesh.extents)) / max(1, int(args.resolution))
    voxelized = mesh.voxelized(pitch)
    filled = voxelized.fill()
    matrix = filled.matrix
    if matrix.size == 0:
        raise ValueError("Voxelization produced an empty shell; inspect the input mesh.")

    wall_iters = max(1, int(round(float(args.wall) / max(float(pitch), 1e-9))))
    interior = ndimage.binary_erosion(matrix, iterations=wall_iters)
    shell = matrix & ~interior
    shelled = trimesh.voxel.ops.matrix_to_marching_cubes(shell, pitch=pitch)
    shelled = repair_mesh(shelled)

    lowest = find_lowest_points(shelled, args.drain_holes)
    if not lowest:
        raise ValueError("No valid lowest points were found for drain-hole placement.")

    radius = max(float(args.drain_diameter) / 2.0, 0.2)
    height = max(float(shelled.extents.max()) * 2.0, 2.0)
    placed = 0
    for idx in lowest:
        center = shelled.vertices[idx].copy()
        hole = trimesh.creation.cylinder(radius=radius, height=height, sections=18)
        hole.apply_translation(np.array([center[0], center[1], center[2] - height / 2.0], dtype=float))
        try:
            shelled = trimesh.boolean.difference([shelled, hole], engine="manifold")
            placed += 1
        except Exception:
            print("  Boolean hole punch unavailable; keeping the hollow shell without drain holes.")
            break

    out = Path(args.out) if args.out else Path(args.model).with_name(Path(args.model).stem + "_hollow.stl")
    save_mesh(shelled, out)
    print(f"Hollowed at ~{args.wall} mm wall (resolution {args.resolution}) -> {out}")
    if placed:
        print(f"Placed {placed} drain hole(s) at the lowest points so trapped material can escape.")
    else:
        print("Drain holes were not placed because the boolean backend was unavailable.")


def cmd_make_printable(args):
    mesh = load_mesh(args.model)
    printable = upgrade_mesh(mesh, size=args.size)
    out = Path(args.out) if args.out else Path(args.model).with_name(Path(args.model).stem + "_printable.stl")
    save_mesh(printable, out)
    rep = mesh_report(printable)
    print_report(rep, "Print-ready model")
    verdict = "BAMBU STUDIO READY" if rep["watertight"] else "SLICE WITH CAUTION (not watertight)"
    print(f" Verdict: {verdict}")
    print(f" Saved -> {out}")


def cmd_convert(args):
    mesh = load_mesh(args.model)
    out = Path(args.out) if args.out else Path(args.model).with_suffix(".stl")
    save_mesh(mesh, out)
    print(f"Exported clean binary STL for Bambu Studio -> {out}")


def main():
    ap = argparse.ArgumentParser(description="ScanForge: real objects -> printable files, one command at a time")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("guide", help="photo capture checklist").set_defaults(func=cmd_guide)

    p = sub.add_parser("scan", help="FULL PIPELINE: photos -> mesh -> STL for Bambu Studio")
    p.add_argument("dir", help="folder of photos")
    p.add_argument("--size", type=float, default=None, help="target longest side in mm")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("check", help="inspect a mesh")
    p.add_argument("model")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("repair", help="repair a mesh")
    p.add_argument("model")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_repair)

    p = sub.add_parser("scale", help="scale longest dimension to mm")
    p.add_argument("model")
    p.add_argument("--size", type=float, required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_scale)

    p = sub.add_parser("hollow", help="shell the model to save filament")
    p.add_argument("model")
    p.add_argument("--wall", type=float, default=2.0, help="wall thickness mm")
    p.add_argument("--resolution", type=int, default=150, help="voxels across longest side")
    p.add_argument("--drain-holes", type=int, default=2, help="number of drain holes to place")
    p.add_argument("--drain-diameter", type=float, default=4.0, help="drain hole diameter in mm")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_hollow)

    p = sub.add_parser("make-printable", help="repair + scale + Bambu Studio bed placement -> STL")
    p.add_argument("model")
    p.add_argument("--size", type=float, default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_make_printable)

    p = sub.add_parser("upgrade", help="alias for make-printable")
    p.add_argument("model")
    p.add_argument("--size", type=float, default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_make_printable)

    p = sub.add_parser("convert", help="re-export as clean binary STL")
    p.add_argument("model")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_convert)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
