# Gereki kütüphaneleri dahil ediyoruz.
import numpy as np
import matplotlib.pyplot as plt
import argparse
import imutils
import cv2

m = []
n= []

def sort_contours(cnts, method="left-to-right"):
	# tersleme değişkenini ve sıralama dizinini başlat
	reverse = False
	i = 0
	# tersine sıralamamız gerekirse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# sınırlama kutusunun x koordinatı yerine y koordinatına göre sıralama yapıyorsak
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1

	# sınırlayıcı kutuların listesini oluşturun ve üstten alta doğru sıralayın
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# sıralanan konturlar ve sınırlayıcı kutular listesini döndürür
	return (cnts, boundingBoxes)

def draw_contour(image, c, i):
	# kontur alanının merkezini hesaplayın.
	print(c)
	M = cv2.moments(c)
	cx = int(M["m10"] / M["m00"])
	cy = int(M["m01"] / M["m00"])
	# Her çerçevenin merkez noktalarını (x -> m, y -> n) dizilere kaydet.
	m.append(cx)
	n.append(cy)
	# Konturun çevresini hesaplıyoruz
	peri = cv2.arcLength(c, True)
	# Köşe sayısını (kordinatlarını) elde ediyoruz
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)
	print(approx)
	if len(approx) == 3:
		shape = "Ucgen"
	elif len(approx) == 4:
		(x, y, w, h) = cv2.boundingRect(approx)
		oran = w / float(h)

		if oran == 1:
			shape = "Kare"
		else:
			shape = "Dikdortgen"
	elif len(approx) == 5:
		shape = "Besgen"
	else:
		shape = "Daire"
	# kontur numaralarını çizin.
	cv2.putText(image, "{}".format(i + 1), (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)
	cv2.circle(image, (cx, cy), 3, (0, 0, 0), -1)
	cv2.drawContours(image, [c], -1, (255, 100, 255), 2)
	cv2.putText(image, shape, (cx-20, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	return image

# bağımsız değişken ayrıştırıcısını oluşturma ve bağımsız değişkenleri ayrıştırma
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-m", "--method", required=True, help="Sorting method")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 11)

thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# Kontur bulma
# Farklı Opencv versiyonları ile uyumlu hale getirme
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# konturları sağlanan yönteme göre sıralayın
(cnts, boundingBoxes) = sort_contours(cnts, method=args["method"])
# (sıralanan) konturların üzerinde döngü oluşturun ve çizin.
for (i, c) in enumerate(cnts):
	draw_contour(image, c, i)

cv2.imshow("Sirali", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

fig = plt.figure()
axes = fig.add_axes([0.1,0.2,0.8,0.5])
axes.plot(m,n,"red",marker = "o",markerfacecolor="blue",markeredgecolor="yellow")
axes.set_xlabel("Y")
axes.set_ylabel("X")
axes.set_title("Çerçevelerin Merkez Noktaları")
plt.show()