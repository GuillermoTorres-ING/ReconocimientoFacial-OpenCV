import cv2
import os

dataPath = 'C:/Users/Guillermo Torres/source/repos/Open/Open/Reconocimiento Facial/Data'
imagePath = os.listdir(dataPath)

reconocedor = cv2.face.LBPHFaceRecognizer_create()

reconocedor.read('faceRecog.xml')

captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#Aqui se inicia la captura de imagen mediante el controlador de camara

clasificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    #Dentro de este Ciclo lo que se hace es cambiarle el color a la ventana para que muestre la imagen
    ret, marco = captura.read()
    if ret == False:
        break
    gris = cv2.cvtColor(marco, cv2.COLOR_BGR2GRAY)
    auxMarco = gris.copy() #Para sacar una captura de cada Frame

    cara = clasificador.detectMultiScale(gris, 1.3, 5)

    for(x, y, w, h) in cara:
        rostro = auxMarco[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (720,720), interpolation = cv2.INTER_CUBIC)
        resultado = reconocedor.predict(rostro)

        cv2.putText(marco, '{}'.format(resultado), (x, y-5), 1, 1.3, (255,0,0),1, cv2.LINE_AA)

        if resultado[1] < 25: #El numero indica el porcentaje de coincidencia entre la imagen y la data para hacer la prediccion
            cv2.putText(marco, '{}'.format(imagePath[resultado[0]]), (x, y-25), 2, 1.1, (0,255,0),1, cv2.LINE_AA)
            cv2.rectangle(marco, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(marco, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255),1, cv2.LINE_AA)
            cv2.rectangle(marco, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('marco', marco)
    k = cv2.waitKey(1)
    if k==27: break

captura.release()
cv2.destroyAllWindows()
