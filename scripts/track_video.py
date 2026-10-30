#!/usr/bin/env python
"""Run a registered detector + Ultralytics tracker on one video (D-028) -- Colab / lab GPU only.

    python scripts/track_video.py --video /content/clip.mp4 \
        --weights /content/drive/MyDrive/uwct/runs/E001/train/weights/best.pt \
        --out /content/drive/MyDrive/uwct/tracks/E001_bytetrack/clip
    python scripts/analyze_tracks.py --tracks /content/drive/MyDrive/uwct/tracks/E001_bytetrack/clip

Writes tracks.csv + tracks_meta.json (docs/PIPELINE.md). Refuses output inside data/raw.
NOT YET RUN on project data: no video exists yet. Exit code: 0 = written, 1 = refused, 2 = usage error.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracking.run import ARCHS, TRACKERS, TrackRunError, track_video
from tracking.schema import TracksError


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="a registered best.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", choices=ARCHS, default="yolo", help="rtdetr for E005 weights")
    parser.add_argument("--tracker", choices=TRACKERS, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--video-id", default=None, help="default: the video file stem")
    parser.add_argument("--synthetic", action="store_true", help="mark a generated or non-project demo clip")
    args = parser.parse_args()

    try:
        meta = track_video(args.video, args.weights, args.out, arch=args.arch, tracker=args.tracker,
                           conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
                           video_id=args.video_id, synthetic=args.synthetic)
    except (TrackRunError, TracksError) as e:
        print(f"TRACKING REFUSED: {e}")
        sys.exit(1)
    print(f"tracked {meta['video_id']}: {meta['n_frames']} frames, fps {meta['fps']} "
          f"({meta['fps_source']}) -> {args.out}")


if __name__ == "__main__":
    main()
