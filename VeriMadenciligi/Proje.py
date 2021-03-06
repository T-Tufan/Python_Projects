import csv
import math
import matplotlib.pyplot as plt

#from sklearn.metrics import  roc_curve



#Roc curve grafiği çizdirmek için aşağıdaki yorum satırlarının, üstteki sklearn ve matplotlib kısımları  kullanılmalı.
#Terminalden sckit kütüphanesi de eklenmeli.Terminale " pip install -U scikit-learn " yazarak kütüphaneyi yükleyebilirsiniz.
#Kütüphane yükledikten sonra 200 mb civarı yer kapladığı için bu koda ekleyemedim.
""" def graph(fpr, tpr, basari_orani):
    plt.plot(fpr, tpr, color='blue', label='Graphs',linestyle='dashed',linewidth=2)
    plt.plot([0, 1], [0, 1], color='purple', linestyle='dashed')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('başarı Oranı:{}'.format(round((basari_orani), 2)))
    plt.show()
"""

def veriSetiYukle(dosya_konum, egitim_oran, egitim_verileri=[], test_verileri=[]):
    with open(dosya_konum, 'rt') as csvDosyası:  # rt okuma ve yazma izinli
        satırlar = csv.reader(csvDosyası)
        tüm_Veriseti = list(satırlar)
        egitim_verileri_uzunluk = len(tüm_Veriseti) * egitim_oran
        test_verileri_uzunluk = len(tüm_Veriseti) * (1 - egitim_oran)
        print("-------Kullanılan Veri Seti------",tüm_Veriseti)
        print("-------Veri Seti Boyutu------",len(tüm_Veriseti))

        for x in range(len(tüm_Veriseti)):
            for y in range(4):
                tüm_Veriseti[x][y] = float(tüm_Veriseti[x][y])
        for k in range(int(egitim_verileri_uzunluk)):
                egitim_verileri.append(tüm_Veriseti[k])
        for j in range(int(test_verileri_uzunluk)):
                test_verileri.append(tüm_Veriseti[j + int(test_verileri_uzunluk)])
        print("-------Eğitim Verileri------", egitim_verileri)
        print("-------Eğitim Verisi Boyutu-------",len(egitim_verileri) )
        print("-------Test Verileri------", test_verileri)
        print("-------Test Verisi Boyutu-----",len(test_verileri))




#1.parametre tüm eğitim verilerini içeriyor.2.parametre her bir eğitim verisini içeriyor.3.parametre test verilerin kaç öznitelikten oluştuğunu içeriyor.
#Bu verisetinde 4 öznitelik var.
#for döngüsü ile her bir test verisi parametre olarak verilen eğitim verisi ile mesafesi ölçülüyor ve mesafe değişkenine atılıyor.
def oklidMesafesi(test_verileri, egitim_verisi, len_testveri):
    mesafe = 0
    for x in range(len_testveri):
        mesafe += pow((test_verileri[x] - egitim_verisi[x]), 2)
    return math.sqrt(mesafe)


def manhattanMesafesi(test_verileri, egitim_verisi, len_testveri):
    mesafe = 0
    for i in range(len_testveri):
            mesafe += (abs(test_verileri[i] - egitim_verisi[i]))
    return mesafe


def minkowskiMesafesi(test_verileri, egitim_verisi, len_testveri):
    p=3
    mesafe=0
    for x in range (len_testveri):
        mesafe +=pow(abs(test_verileri[x]-egitim_verisi[x]),p)
        mesafe +=pow(mesafe,(1/p))
    return mesafe



#İlk olarak parametre olarak verilen hesaplama türüne göre test verilerinin eğitim verilerine göre mesafelerine bakılır.mesafe değişkenine atılır.
#Daha sonra her test verisinin hangi eğitim verisine ne kadar uzaklıkta olduğu mesafesi ile beraber mesafeler dizisine atılır.
#mesafeler.sort ile en kısa mesafeli olan egitim verileri mesafeler dizisinde en başa gelecek şekilde sıralanır.
#En alttaki döngüde parametre olarak gelecek k değerine göre mesafeler dizisindeki ilk k eleman alınır ve en_yakinler dizisine atılır.En yakinler dizisi döndürülür.
def en_yakın_degerler(egitim_verileri, test_verisi,hesaplama_türü,k):
    mesafeler = []
    en_yakinlar = []
    test_veri_boyutu = len(test_verisi) - 1
    for x in range(len(egitim_verileri)):
        mesafe = hesaplama_türü(test_verisi, egitim_verileri[x], test_veri_boyutu)
        mesafeler.append((egitim_verileri[x], mesafe))#distances dizisine eğitim verisini ve ona ait uzunlukları sırayla(x değerine bağlı) yazar.
    mesafeler.sort(key=lambda elem: elem[1])
    for x in range(k):
        en_yakinlar.append(mesafeler[x][0])
    return en_yakinlar


######################
# PREDICTED RESPONSE #
######################

import operator

#En son test verilerinin sınıflarının tutulacağı bir veri_sınıfı dizisi oluşturuldu.Daha sonra for döngüsünün k değeri kadar dönmesi için en_yakınlar değerininin
#uzunluğu alındı.deger değişkenine en yakın k adet verinin son elemanı yani sınıfı atılıyor.Burada en yakın mesafenin sınıfı deger değiskenine atılıyor.
# Diger en yakın degerlerin sınıflarıda bu degerle aynı ise sayac her defasında  1 arttırılıyor.Bu sayede en yakın verilerin sınıflarını gruplandırıp saymış oluyoruz.
def Sınıflandırma(en_yakınlar):
    veri_sınıfı = {}
    for x in range(len(en_yakınlar)):
        deger = en_yakınlar[x][-1]
        if deger in veri_sınıfı:
            veri_sınıfı[deger] += 1
        else:
            veri_sınıfı[deger] = 1
    veri_sınıfı = sorted(veri_sınıfı.items(), key=operator.itemgetter(1), reverse=True)
    return veri_sınıfı[0][0]


def basari_orani(test_veri, tahminler):
    dogru_tahmin = 0
    for x in range(len(test_veri)):
        if test_veri[x][-1] in tahminler[x]:
            dogru_tahmin = dogru_tahmin + 1
    return (dogru_tahmin / float(len(test_veri)) * 100)

def tüm_fonksiyonlar(k,p) :
    k_dizisi = [1, 2, 3, 4, 5, 10, 15 , 25, 35, 45]
    hesaplama_tipi = [oklidMesafesi, manhattanMesafesi, minkowskiMesafesi]
    egitim_verileri = []
    test_verileri = []
    real_sonuclar = []
    oran = 0.60
    veriSetiYukle(r'../data/data_banknote_authentication.csv', oran, egitim_verileri, test_verileri)
    print('Train Verileri Sayısı: ' + repr(len(egitim_verileri)))
    print('Test Verileri Sayısı: ' + repr(len(test_verileri)))

    tahminler = []
    print("------------------------------------------\n----------k eşittir = ", k_dizisi[k],
          " değeri için---------\n--------------------------------------")
    for x in range(len(test_verileri)):
        yakın_mesafeler = en_yakın_degerler(egitim_verileri, test_verileri[x], hesaplama_tipi[p], k_dizisi[k], )
        sınıf = Sınıflandırma(yakın_mesafeler)
        tahminler.append(sınıf)
    basariOrani = basari_orani(test_verileri, tahminler)


    print("Tahmini Sonuçlar", tahminler)
    for i in range(len(test_verileri)):
        real_sonuclar.append(test_verileri[i][4])
    print("Real Sonuçlar", real_sonuclar)
    print('Başarı Oranı: ' + repr(basariOrani) + '%')

    #Real ve tahminler dizisindeki string degerler int veriye dönüstürülür.
    for i in range(0, len(real_sonuclar)):
        real_sonuclar[i] = int(real_sonuclar[i])
    for i in range(0, len(tahminler)):
        tahminler[i] = int(tahminler[i])

    #Programın basarı oranı hesaplatılır.Roc curve çizdirmek için alttaki kod satırlarının açılması gerekiyor.
    """
    fpr, tpr, thresholds = roc_curve(real_sonuclar, tahminler)
    plt.figure(figsize=(3, 3))
    graph(fpr, tpr, basariOrani)
    plt.show()
    """

#Kullanğım verisetine göre program %100 e yakın başarı oranı vermektedir.K değerinin yüksek verilmesi durumunda oran %90 lara düşmektedir.
#Deneme aşamasında kullandığım diğer verisetlerinde bu oran %75 ile %95 arasında değişmekteydi.
def kullanıcı_tarafi():
    # prepare data
  while True:
    p=int(input("P değeri giriniz"))
    print("Girdiğiniz P değeri : ", p)
    p=p-1
    k=int(input("K değeri giriniz"))
    print("Girdiğiniz K değeri : ",k)
    k=k-1

    if p==0 :
        print("---------Öklid Fonksiyonu Sonuçları")
    elif p==1 :
        print("---------Manhattan Fonksiyonu Sonuçları")
    elif p==2 :
        print("---------Minkowski Fonksiyonu Sonuçları")
    else :
        print("Geçerli bir p değeri giriniz")
        continue;
    tüm_fonksiyonlar(k,p)
    print("Devam etmek için değer giriniz\n")

kullanıcı_tarafi()

