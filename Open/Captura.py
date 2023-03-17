import cv2
import os
import imutils

#nombrePersona = ''
dataPath = 'C:/Users/Guillermo Torres/source/repos/Open/Open/Reconocimiento Facial/Data'
#DataPath es la ruta de almacenamiento de la data

nombrePersona = input('Ingresa el nombre: ')
personaPath = dataPath + '/' + nombrePersona
if not os.path.exists(personaPath):
    print('Carpeta Creada exitosamente', personaPath)
    os.makedirs(personaPath)

 #Aqui se verifica mediante un if con OS si la carpeta existe, si esta no existe se crea una nueva

captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#Aqui se inicia la captura de imagen mediante el controlador de camara

clasificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cont = 0
while True:
    #Dentro de este Ciclo lo que se hace es cambiarle el color a la ventana para que muestre la imagen
    ret, marco = captura.read()
    if ret == False:
        break
    marco = imutils.resize(marco, width=480) #Para redimensionar la imagen
    gris = cv2.cvtColor(marco, cv2.COLOR_BGR2GRAY)
    auxMarco = marco.copy() #Para sacar una captura de cada Frame

    cara = clasificador.detectMultiScale(gris, 1.3, 5)

    for(x, y, w, h) in cara:
        cv2.rectangle(marco, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rostro = auxMarco[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (720,720), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(personaPath + '/rostro_{}.jpg'. format(cont), rostro)
        cont = cont + 1

    cv2.imshow('marco', marco)

    k = cv2.waitKey(1)
    if k == 27 or cont>=400:
        break

captura.release()
cv2.destroyAllWindows()
