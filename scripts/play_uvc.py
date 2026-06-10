import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import threading


def open_capture(index, width=None, height=None, use_mjpg=False, buffer_size=None, fps=None, use_gstreamer=False):
    if use_gstreamer:
        # build a simple pipeline that tries to minimize buffering
        pipeline = (
            f'v4l2src device=/dev/ur5e/dji_osmo_action5pro_0 '
            f'! video/x-raw,format=YUY2,width={width},height={height},framerate={int(fps)}/1 '
            f'! queue max-size-buffers=1 leaky=downstream '
            f'! videoconvert ! appsink drop=true max-buffers=1 sync=false'
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        return None
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps:
        try:
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        except Exception:
            pass
    if use_mjpg:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
    if buffer_size is not None:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))
        except Exception:
            pass
    return cap


def make_placeholder(width, height, text="No Signal"):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, text, (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


class LatestFrame:
    """Background thread that always keeps the latest frame (drop older frames).

    Use this to reduce capture-to-display latency.
    """
    def __init__(self, src=0, width=640, height=480, use_gstreamer=False, use_mjpg=True, buffer_size=1, fps=30):
        if use_gstreamer:
            self.cap = open_capture(src, width, height, use_mjpg=False, buffer_size=buffer_size, fps=fps, use_gstreamer=True)
        else:
            self.cap = open_capture(src, width, height, use_mjpg=use_mjpg, buffer_size=buffer_size, fps=fps)

        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            if self.cap is None:
                time.sleep(0.01)
                continue
            # use grab/retrieve to minimize blocking
            if not self.cap.grab():
                time.sleep(0.005)
                continue
            ret, f = self.cap.retrieve()
            if not ret or f is None:
                continue
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)
        if self.cap is not None:
            self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="Read one UVC device and display it with latency-optimized options.")
    parser.add_argument("--device", type=int, default=0, help="摄像头设备索引 (默认: 0)")
    parser.add_argument("--width", type=int, default=1280, help="输出宽度 (默认: 640)")
    parser.add_argument("--height", type=int, default=720, help="输出高度 (默认: 480)")
    parser.add_argument("--window", type=str, default="UVC View", help="显示窗口名")
    parser.add_argument("--fps", type=float, default=30.0, help="显示速率上限 (默认: 30)")
    # latency-related switches (defaults chosen for low latency)
    parser.add_argument("--use-mjpg", action="store_true", default=True, help="尝试使用 MJPG 编码以减少延迟 (默认: True)")
    parser.add_argument("--no-mjpg", dest="use_mjpg", action="store_false", help="禁用 MJPG")
    parser.add_argument("--buffer-size", type=int, default=1, help="设置 CAP_PROP_BUFFERSIZE (默认:1)")
    parser.add_argument("--use-thread", action="store_true", default=True, help="启用后台线程读取只保留最新帧 (默认: True)")
    parser.add_argument("--no-thread", dest="use_thread", action="store_false", help="禁用后台线程读取")
    parser.add_argument("--gstreamer", action="store_true", default=False, help="使用 GStreamer pipeline (可选，默认: False)")
    parser.add_argument("--disable-auto-exposure", action="store_true", default=True, help="尝试禁用自动曝光以减少延迟 (默认: True)")
    parser.add_argument("--exposure", type=float, default=None, help="若禁用自动曝光，可设置固定曝光值")
    args = parser.parse_args()

    target_w = args.width
    target_h = args.height
    delay = int(1000 / max(1.0, args.fps))
    capture_dir = Path("./capture")

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    # if using thread, create LatestFrame; else open capture directly
    reader = None
    cap = None

    try:
        if args.use_thread:
            reader = LatestFrame(src=args.device, width=target_w, height=target_h, use_gstreamer=args.gstreamer,
                                 use_mjpg=args.use_mjpg, buffer_size=args.buffer_size, fps=args.fps)
            # attempt to adjust camera controls via the underlying cap if available
            cap_for_ctrl = reader.cap
        else:
            cap = open_capture(args.device, target_w, target_h, use_mjpg=args.use_mjpg, buffer_size=args.buffer_size, fps=args.fps, use_gstreamer=args.gstreamer)
            cap_for_ctrl = cap

        if cap_for_ctrl is None:
            print(f"无法打开设备 {args.device}")
            sys.exit(1)

        # try to disable auto exposure / auto white balance for lower jitter
        if args.disable_auto_exposure:
            try:
                # 1 may mean manual for some backends
                cap_for_ctrl.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                if args.exposure is not None:
                    cap_for_ctrl.set(cv2.CAP_PROP_EXPOSURE, float(args.exposure))
            except Exception:
                pass
            try:
                cap_for_ctrl.set(cv2.CAP_PROP_AUTO_WB, 0)
            except Exception:
                pass

        while True:
            if reader is not None:
                frame = reader.read()
            else:
                ret, frame = cap.read()
                if not ret:
                    frame = None

            if frame is None:
                frame = make_placeholder(target_w, target_h, f"Device {args.device} - No Signal")
            else:
                frame = cv2.resize(frame, (target_w, target_h))
                # cv2.putText(frame, f"Device:{args.device}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(args.window, frame)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('c'):
                capture_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = capture_dir / f"capture_{timestamp}_{int(time.time_ns() % 1_000_000_000):09d}.jpg"
                if cv2.imwrite(str(filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                    print(f"已保存截图: {filename}")
                else:
                    print(f"截图保存失败: {filename}", file=sys.stderr)
            if key == ord('q') or key == 27:  # q 或 ESC 退出
                break

    finally:
        if reader is not None:
            reader.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
