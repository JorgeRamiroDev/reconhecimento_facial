import cv2
import face_recognition as fr 

############## Informando as pessoas que devem ser reconhecidas 
ivan = fr.load_image_file('playground\\facial_database\\WIN_20240619_10_25_42_Pro.jpg')
ivan_encoding = fr.face_encodings(ivan)[0]

# elon = fr.load_image_file('fotos/Elon.jpg')
# elon_encoding = fr.face_encodings(elon)[0]

# tony = fr.load_image_file('fotos/tonyStark.jpg')
# tony_encoding = fr.face_encodings(tony)[0]

# peter = fr.load_image_file('fotos/peterparker.jpg')
# peter_encoding = fr.face_encodings(peter)[0]

encodings = [
    (ivan_encoding, 'Ivan'), 
    # (elon_encoding, 'Elon Musk'),
    # (tony_encoding, 'Tony Stark'), 
    # (peter_encoding, 'Peter Parker')
    ]

# Abre a webcam
# Digite o número referente a sua webcam 
cap = cv2.VideoCapture(1)

while True:
    # Captura um frame da webcam
    ret, frame = cap.read()

    # Localiza os rostos presentes na imagem
    
    face_locations = fr.face_locations(frame)
    
    # Para cada rosto encontrado faça
    for face_location in face_locations:
        # Pega as coordenadas do rosto
        top, right, bottom, left = face_location

        # Extrai a codificação do rosto
        face_encoding = fr.face_encodings(frame, [face_location])[0]

        # Compara a codificação do rosto com os encoders carregados
        for desconhecido_encoding, name in encodings:
            # Desenha um retangulo em todos os rostos que encontrar 
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            match = fr.compare_faces([desconhecido_encoding], face_encoding)[0]

            # Se houver uma correspondência, desenha um retângulo verde em volta do rosto e exibe o nome da pessoa
            if match:    
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Exibe o frame capturado
    cv2.imshow('Webcam', frame)

    # letra ESQ sai do laço
    key = cv2.waitKey(1)
    # tecla ESC sai do laço
    if key == 27:
        break
    

# Libera a webcam e fecha as janelas
cap.release()
cv2.destroyAllWindows()