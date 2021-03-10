import cv2, os


class Meriti_bot:
    def __init__(self):
        pass

    def buscar_partes(self, name, path):
        for root, dirs, files in os.walk(path):
            if (name in files) or (name in dirs):
                return os.path.join(root, name)
        return self.buscar_partes(name, os.path.dirname(path))

    def reconhecer_faces_fotos(self, url):
        cv2path = os.path.dirname(cv2.__file__)
        reconhecimento_facial_xml = self.buscar_partes('haarcascade_frontalface_alt2.xml', cv2path)
        clf = cv2.CascadeClassifier(reconhecimento_facial_xml)
        imagem = cv2.imread(url)
        imagem_cinza = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
        rostos = clf.detectMultiScale(imagem_cinza)
        for (x, y, largura, altura) in rostos:
            cv2.rectangle(imagem, (x,y), (x+largura , y+altura),(0,255,0), 2)
            cv2.imshow('Faces', imagem)
            cv2.waitKey()
        cv2.destroyAllWindows()


    def recohecer_faces_video_webcam(self):
        Cor_retangulo = (255, 0, 0)
        STROKE = 2
        cv2path = os.path.dirname(cv2.__file__)

        reconhecimento_facial_xml = self.buscar_partes('haarcascade_frontalface_alt2.xml', cv2path)
        clf = cv2.CascadeClassifier(reconhecimento_facial_xml)
        cap = cv2.VideoCapture(0)

        while (not cv2.waitKey(20) & 0xFF == ord('q')):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = clf.detectMultiScale(gray)
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), Cor_retangulo, STROKE)
            cv2.imshow('frame', frame)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    bot = Meriti_bot()
    bot.recohecer_faces_video_webcam()
