import cv2
import time
from cvzone.HandTrackingModule import HandDetector

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

rastreador = HandDetector(detectionCon=0.8, maxHands=2)

VENTANA = "proyecto 4 - IA"
cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)

# Para evitar que cierre por un falso positivo instantáneo
ok_contador = 0
OK_FRAMES_NEcesarios = 8   # ~8 frames seguidos haciendo OK

while True:
    exito, imagen = webcam.read()
    if not exito:
        break

    imagen = cv2.resize(imagen, (1280, 720))

    manos, imagen_manos = rastreador.findHands(imagen, draw=True, flipType=False)

    ok_detectado = False

    if manos:
        for mano in manos:
            lm = mano["lmList"]  # 21 puntos [x,y,z]

            # punta pulgar = 4, punta índice = 8
            x4, y4, _ = lm[4]
            x8, y8, _ = lm[8]

            # Distancia entre pulgar e índice
            dist, info, _ = rastreador.findDistance((x4, y4), (x8, y8), imagen_manos)

            # Umbral: si están muy cerca => gesto OK
            # (Ajusta 35-55 según tu cámara/distancia)
            if dist < 45:
                ok_detectado = True
                break

    # Si detectó OK varios frames seguidos, confirmamos
    if ok_detectado:
        ok_contador += 1
        cv2.putText(imagen_manos, "OK", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    else:
        ok_contador = 0

    cv2.imshow(VENTANA, imagen_manos)

    # Cerrar si le das a la X
    if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
        break

    # Cerrar con cualquier tecla REAL (filtrada)
    k = cv2.waitKey(1) & 0xFF
    if k not in (255, 0):
        break

    # Cerrar si mantienes el gesto OK
    if ok_contador >= OK_FRAMES_NEcesarios:
        cv2.putText(imagen_manos, "OK - Cerrando...", (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow(VENTANA, imagen_manos)
        cv2.waitKey(700)  # espera 0.7s para que veas el mensaje
        break

webcam.release()
cv2.destroyAllWindows()
