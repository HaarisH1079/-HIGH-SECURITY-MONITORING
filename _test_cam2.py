import cv2, time, sys

results = []
for backend_name, backend in [('CAP_ANY', cv2.CAP_ANY), ('CAP_DSHOW', cv2.CAP_DSHOW), ('CAP_MSMF', cv2.CAP_MSMF)]:
    cap = cv2.VideoCapture(2, backend)
    opened = cap.isOpened()
    if opened:
        time.sleep(0.5)
        ret, f = cap.read()
        shape = f.shape if (ret and f is not None) else None
        results.append(f"{backend_name}: opened={opened}, read={ret}, shape={shape}")
    else:
        results.append(f"{backend_name}: FAILED TO OPEN")
    cap.release()
    time.sleep(0.3)

for r in results:
    print(r)
