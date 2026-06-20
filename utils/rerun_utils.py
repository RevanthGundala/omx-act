"""Rerun visualization helpers for OMX eval scripts."""

from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

from utils.config import CAMERAS, JOINT_NAMES

RRD_DIR = Path("outputs/rerun")


def build_joint_blueprint(
    joint_names: list[str] | None = None,
    has_camera: bool = True,
    camera_primary: bool = False,
    camera_names: list[str] | None = None,
) -> rrb.Horizontal | rrb.Vertical | rrb.Tabs:
    joint_names = joint_names or JOINT_NAMES
    camera_names = camera_names or (list(CAMERAS.keys()) if has_camera else [])

    joint_views = [
        rrb.TimeSeriesView(name=name, contents=[f"joints/{name}/**"])
        for name in joint_names
    ]
    cam_views = [
        rrb.Spatial2DView(name=cam, contents=[f"camera/{cam}/**"])
        for cam in camera_names
    ]
    cam_panel = rrb.Vertical(*cam_views) if cam_views else None

    if has_camera and camera_primary and cam_panel is not None:
        cam_tab = rrb.Horizontal(*cam_views, name="Cameras") if len(cam_views) > 1 else cam_views[0]
        joint_tab = rrb.Vertical(*joint_views, name="Joints")
        return rrb.Tabs(cam_tab, joint_tab)
    if has_camera and cam_panel is not None:
        return rrb.Horizontal(
            cam_panel,
            rrb.Vertical(*joint_views),
            column_shares=[1, 2],
        )
    return rrb.Vertical(*joint_views)


def init_rerun(
    name: str,
    joint_names: list[str] | None = None,
    has_camera: bool = True,
    camera_primary: bool = False,
    save_rrd: bool = True,
    camera_names: list[str] | None = None,
) -> Path | None:
    blueprint = build_joint_blueprint(joint_names, has_camera, camera_primary, camera_names)
    rr.init(name)

    rrd_path = None
    if save_rrd:
        RRD_DIR.mkdir(parents=True, exist_ok=True)
        rrd_path = RRD_DIR / f"{name}.rrd"
        rr.set_sinks(
            rr.GrpcSink(),
            rr.FileSink(str(rrd_path)),
        )
        rr.spawn()
    else:
        rr.spawn()

    rr.send_blueprint(blueprint)
    return rrd_path
