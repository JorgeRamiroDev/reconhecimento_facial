from deepface import DeepFace


img1 = "playground\\facial_database\\WIN_20240619_10_25_42_Pro.jpg"
montedeimg = "playground\\facial_database"

def acha_o_cara(img1_path,img2_path):
    output = DeepFace.find(img1_path,img2_path, detector_backend="opencv")
    print(output)
    verification = output['distance']
    if verification:
       print('achei')
   


acha_o_cara(img1,montedeimg)