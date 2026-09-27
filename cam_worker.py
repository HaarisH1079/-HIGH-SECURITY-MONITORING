"""
cam_worker.py - Isolated camera capture process.

Runs as a child process to capture frames from a camera index that conflicts
with other cameras under Windows MSMF. Sends frames as raw bytes via stdout.

Protocol (stdout, binary):
  Each frame: [4-byte little-endian frame_size][frame_bytes_as_jpeg]
  On failure:  [4-byte 0x00000000] = no frame available yet
"""
import sys
import cv2
import struct
import time

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: cam_worker.py <camera_index>\n")
        sys.exit(1)

    index = int(sys.argv[1])
    sys.stderr.write(f"[cam_worker] Starting for camera index {index}\n")
    sys.stderr.flush()

    cap = None
    fail_count = 0

    def open_camera():
        # Try MSMF first (only camera on this process, so no conflict)
        for backend in [cv2.CAP_MSMF, cv2.CAP_ANY]:
            c = cv2.VideoCapture(index, backend)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                sys.stderr.write(f"[cam_worker] Cam {index} opened (backend={backend})\n")
                sys.stderr.flush()
                return c
        sys.stderr.write(f"[cam_worker] WARNING: Could not open camera {index}\n")
        sys.stderr.flush()
        return None

    cap = open_camera()

    # Use binary stdout for frames
    stdout = sys.stdout.buffer

    while True:
        if cap is None or not cap.isOpened():
            time.sleep(1)
            fail_count += 1
            if fail_count > 5:
                sys.stderr.write(f"[cam_worker] Retrying cam {index}...\n")
                sys.stderr.flush()
                cap = open_camera()
                fail_count = 0
            # Send zero-size frame to signal not ready
            stdout.write(struct.pack('<I', 0))
            stdout.flush()
            continue

        ret, frame = cap.read()
        if ret and frame is not None:
            fail_count = 0
            frame = cv2.flip(frame, 1)
            # Encode as JPEG for compact transfer
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                data = buf.tobytes()
                stdout.write(struct.pack('<I', len(data)))
                stdout.write(data)
                stdout.flush()
            else:
                stdout.write(struct.pack('<I', 0))
                stdout.flush()
        else:
            fail_count += 1
            if fail_count > 30:
                sys.stderr.write(f"[cam_worker] Cam {index} lost frames. Reconnecting...\n")
                sys.stderr.flush()
                cap.release()
                cap = open_camera()
                fail_count = 0
            stdout.write(struct.pack('<I', 0))
            stdout.flush()

if __name__ == '__main__':
    main()
