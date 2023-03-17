import cv2
import os
import numpy as np

dataPath = 'C:/Users/Guillermo Torres/source/repos/Open/Open/Reconocimiento Facial/Data'

personas = os.listdir(dataPath)
print('lista de personas: ', personas)

labels = []
rostrosData = []
label = 0

for nameDir in personas:
    personaPath = dataPath + '/' + nameDir
    print('Analizando...')

    for fileName in os.listdir(personaPath):
        print('Rostros... ', nameDir + '/'+ fileName)
        labels.append(label)

        rostrosData.append(cv2.imread(personaPath + '/'+ fileName, 0))
        imagen = cv2.imread(personaPath + '/'+ fileName, 0)

        #cv2.imshow('imagen', imagen)
        #cv2.waitKey(10)

    label = label + 1
#cv2.destroyAllWindows()


#print('labels = ', labels)
#print('Numero de Etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
#print('Numero de Etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

reconocedor = cv2.face.LBPHFaceRecognizer.create()

print('Entrenando... ')
reconocedor.train(rostrosData, np.array(labels))

reconocedor.write('faceRecog.xml')
print('Guardado...')