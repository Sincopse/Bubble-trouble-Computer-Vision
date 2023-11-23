# trabalhamos aqui
#nao
#sim
#ok
# temos codigo feito do stor certo?
#i guess foi oque estivemos a fazer as ultimas aulas
# eu n estive 💀
#vai ao git e da download, é só as pares de deteção de objetos
#vai tu, n eras tu que querias fazer código (n sei qual é o ficheiro deixa ver)
#eu tenho o codigo do git aqui
#tás a trollar
#queres que eu mande? ou que comece a fazer ja?
# mete aqui em baixo sfv
#como queres que eu meta aqui em baixo xD
# ctrl + c ctrl + v wym, é só 1 ficheiro certo?
#sao 6 e um deles percisa de 1biblioteca nova que estou a instalar
# 6 ficheiros para uma unica deteção de objetos?
#fds anda ca
# anda tu
#:-: pera eu estou a acabar de descarregar umas bibliotecas, queres os codigos para instalares?
# só é preciso do cv2 supostamente, eu tou a rodar o de face detection de boas
#ve ai o 10.3 ta na pasta coisas da aula
# este aqui funciona
#para mim esse e oque nao funciona
#ve aquela pasta no canto superior esquerdo "coisas da aula" la tens o 10.3
# já tou a ver, é pra escolher qual quero?
#tipo n n escolhemos usar o que roda a 4 fps?
# pensei nisso, ent pq mostras os outros dois exercicios?
#pq tu pediste para mandar os ficheiros da aula
# pedi? eu pedi para mandres o código do que nós iamos fazer
#linha 8 a 13
# ent, eu pedi para meteres 1 deles neste ficheiro, como eu fiz aqui em baixo
#ta esquece ent os outros 2 e so o 10.3 que percisamos
# deixa ver os outros dois primeiro
#o 10.1 n consigo rodar mas o 10.2 e oque deteta pessoas na rua
#ok
#o 10.1 tbm n tem nada de jeito pra detetar por default (pessoas, caras, matriculas)
# podemos é treinar para detetar por exemplo um telemóvel
#da para fazer ele detetar so o telemovel?
#sim, se o conseguirmos treinar
#o stor ja espelicou como treinamos?
#acho que sim. edit:afinal não
#

import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Open camera
cap = cv2.VideoCapture(0)  # Change to the appropriate camera index if needed

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detecting objects
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('YOLO Object Detection', frame)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
