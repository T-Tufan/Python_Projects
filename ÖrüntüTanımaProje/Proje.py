import csv
import math
import statistics as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt

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


def Pca():
  print("-----------------------------------\n      ---------  PCA  ---------     \n-----------------------------------")
  print("Kullanılan verisetinin ilk hali : ")
  veri = pd.read_csv("../data/pca_covid19.csv")#Kullanılacak olan dataset verileri programa veriliyor.

  print(veri)#Pca ya uğramamış dataset verileri ekrana yazdırılıyor.

  x = veri.iloc[:, 0:5].values
  np.set_printoptions(formatter={'float_kind':'{:f}'.format}) # uzun float sayıları daha kısa göstermeyek için kullanılan kod.

  x = StandardScaler().fit_transform(x)
  # Yeni verisetinde kaç öznitelik sütunu olacağını giriliyor.
  pca = PCA(n_components=4)
  pct = pca.fit_transform(x)
  # Yeni verisetindeki öznitelik sütunlarındaki değerler ve öznitelik sütunu isimleri birleştiriliyor.
  principal_df = pd.DataFrame(pct, columns=['att1', 'att2','att3','att4'])
  #Öznitelik sütunarı ve Sınıf sütunu birleştiriliyor.
  finaldf = pd.concat([principal_df, veri[['Durum']]], axis=1)

  print("\nYukarıdaki 5 öznitelik bilgisine sahip veriseti pca kullanılarak aşağıdaki 4 öznitelik bilgisi içeren verisetine dönüştürülmüştür.")
  #Yeni oluşturulan veriseti ekrana bastırılır.
  print(finaldf)

#X için Gauss olasılık dağılım fonksiyonunu hesaplanıyor
def gauss(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev * stdev)) * exponent

def NaiveBayes(oran):
    egitim_verileri = []
    test_verileri = []
    #Veriseti okuma işlemleri burada yapılıyor.
    veriseti, egitim_veri, test_veri = BayesVeriSeti(r'../data/covid19.csv', oran, egitim_verileri, test_verileri)
    #Veriseti ile ilgili genel bilgiler veriliyor.
    print("dataset boyutu : ", len(veriseti))
    print("test veri boyutu : ", len(test_veri))
    print("eğitim veri boyutu : ", len(egitim_veri))

    sınıflar = []
    for i in veriseti: #veriseti
        if (i[-1] not in sınıflar):#veriseti dizisindeki son sütun değerleri ekleniyor.Eğer o deger dizide var ise ikinci defa eklenmiyor.
            sınıflar.append(i[-1])  #dataset içindeki verilerin son sütunlarındaki değerleri sınıflar dizisine atıyoruz.
    sınıf = {}  # tüm sınıf değerlerini içeren dizi (verisetindeki sıraya göre)
    hesaplamalar = {}  # Verilere ait aritmetik ortalama ve standart sapma bilgileri bu diziye atılıyor.
    sınıf_ihtimalleri = {}  # Test verilerine bakılarak hangi sınıfa ait olabileceği bilgisi bu diziye atılır.
    #verilen test verilerinin ilgili sınıf değerlerine denk gelme oranı
    for i in sınıflar:
        sınıf[i] = []
        hesaplamalar[i] = []
        sınıf_ihtimalleri[i] = 1

    #her sınıf değerinin ait olduğu satır o sınıf değerine eklenir.
    #Verisetimizde iki adet sınıf (0,1) olduğu için bizimde iki boyutlu dizimiz oluşturuluyor.
    for i in sınıflar:
        for row in egitim_veri: #eğitim verilerindeki satırlara tek tek bakılıyor.
            if row[-1] == i:#İlgili satırın son sütununa bakılır.i değerine eşit ise satır, classdict[i] dizisine atılır.
                sınıf[i].append(row[:-1])
        if(i == "1"):
            print("1 sınıfına ait satırlar :", sınıf[i]) #"1" sınıfına ait satırlar ekrana bastırılır.
        if(i == "0"):
            print("0 sınıfına ait satırlar :",sınıf[i]) #"0" sınıfına ait satırlar ekrana bastırılır.
    for sınıf_degeri, oznitelikler in sınıf.items():#Bu kısımda sınıflar ve öznitelik bilgileri ayrı değişkenlere atılır.
        #print("sınıf : ",classval)
        #print("Öznitelik satırları : ", *datt)
        for col in zip(*oznitelikler):
            # Bu kısımda ilgili verisetindeki satırlara ait her öznitelik sütunu gruplandrılır.
            # Bu kısımda ilk olarak 1 sınıfına ait satırların sütun değerleri kendi aralarında gruplandırılır.
            # Örnek olarak
            # 5 adet öznitelik değeri var.Sigara kullanımı,yaşı,tansiyon değerleri,kilolar ve cinsiyet...
            #print("sütun : ",col)
            hesaplamalar[sınıf_degeri].append((st.mean(col), st.stdev(col)))
            #st.mean() fonksiyonu ortalama hesaplaması yapar.İlgili verilerin ortalamasını hesaplar.
            #st.stdev() fonksiyonu standart sapma hesaplaması yapar.
            #Bu bölümde gruplandırılan öznitelik değerlerinin aritmetik ortalaması ve standart sapması hesaplanır.
            #hesaplamalar  dizisine atılır.
        #print("hesaplamalar ",hesaplamalar[sınıf_degeri])
    sayac = 0  # En sonunda doğru tahmin edilen test verisi sayılarını sayarken sayaç kullanılacak.
    for row in test_veri:#Kullanılacak test verilerindeki satırlar tek tek döndürülür.
        for i in sınıflar:
            sınıf_ihtimalleri[i] = 1
        for sınıf_degeri, oznitelikler in hesaplamalar.items():
            #Her sınıfa ait verilerin aritmetik ortalaması ve standart sapması classval ve datt değişkenlerine atılır..
            for i in range(len(row[:-1])):#satırın son elemanı yani sınıfı ele alınır.
                ort, std_sapma = oznitelikler[i]
                x = row[i]

                sınıf_ihtimalleri[sınıf_degeri] *= gauss(x, ort, std_sapma)  #
        print(row, " test verisi için sonuçlar : ", sınıf_ihtimalleri)
        # Programın basarı oranı
        esik_deger = 0
        durum = 0
        for snf, dgr in sınıf_ihtimalleri.items():#1 ve 0 sınıflarına ait mesafeler degerlere atılır.c sınıfı d mesafeti alır.
            if dgr > esik_deger: #mesafe değeri 0 'dan büyükse deger
                esik_deger = dgr
                durum = snf

        if row[-1] == durum:
            sayac += 1
    np.set_printoptions(formatter={'float_kind': '{:f}'.format})
    basari_orani = sayac / float(len(test_veri)) * 100
    print("Başarı Oranı : ", round(basari_orani,2),"%")
def BayesVeriSeti(dosya_konum, egitim_oran, egitim_verileri=[], test_verileri=[]):
    with open(dosya_konum, 'rt') as csvDosyası:  # rt okuma ve yazma izinli
        satırlar = csv.reader(csvDosyası)
        tüm_Veriseti = list(satırlar)
        egitim_verileri_uzunluk = len(tüm_Veriseti) * egitim_oran
        test_verileri_uzunluk = len(tüm_Veriseti) * (1 - egitim_oran)
        for x in range(len(tüm_Veriseti)):
            for y in range(5):
                tüm_Veriseti[x][y] = float(tüm_Veriseti[x][y])
        for k in range(int(egitim_verileri_uzunluk)):
                egitim_verileri.append(tüm_Veriseti[k])
        for j in range(int(test_verileri_uzunluk)):
                test_verileri.append(tüm_Veriseti[j + int(test_verileri_uzunluk)])

        return tüm_Veriseti, egitim_verileri, test_verileri
def veriSetiYukle(dosya_konum, egitim_oran, egitim_verileri=[], test_verileri=[]):
    with open(dosya_konum, 'rt') as csvDosyası:  # rt okuma ve yazma izini verir
        satırlar = csv.reader(csvDosyası)
        tüm_Veriseti = list(satırlar)
        print(tüm_Veriseti)
        egitim_verileri_uzunluk = len(tüm_Veriseti) * egitim_oran
        test_verileri_uzunluk = len(tüm_Veriseti) * (1 - egitim_oran)
        print("-------Kullanılan Veri Seti------",tüm_Veriseti)
        print("-------Veri Seti Boyutu------",len(tüm_Veriseti))

        for x in range(len(tüm_Veriseti)):
            for y in range(5):
                tüm_Veriseti[x][y] = float(tüm_Veriseti[x][y])
        for k in range(int(egitim_verileri_uzunluk)):
                egitim_verileri.append(tüm_Veriseti[k])
        for j in range(int(test_verileri_uzunluk)):
                test_verileri.append(tüm_Veriseti[j + int(test_verileri_uzunluk)])
        print("-------Eğitim Verileri------", egitim_verileri)
        print("-------Eğitim Verisi Sayısı-------",len(egitim_verileri) )
        print("-------Test Verileri------", test_verileri)
        print("-------Test Verisi Sayısı-----",len(test_verileri))
        return  tüm_Veriseti , egitim_verileri , test_verileri


#1.parametre tüm eğitim verilerini içeriyor.2.parametre her bir eğitim verisini içeriyor.3.parametre test verilerin kaç öznitelikten oluştuğunu içeriyor.
#Bu verisetinde 4 öznitelik var.
#for döngüsü ile her bir test verisi parametre olarak verilen eğitim verisi ile mesafesi ölçülüyor ve mesafe değişkenine atılıyor.
def oklidMesafesi(test_verileri, egitim_verisi, len_testveri):
    mesafe = 0
    for x in range(len_testveri):
        mesafe += pow((float(test_verileri[x]) - float(egitim_verisi[x])), 2)
    return math.sqrt(mesafe)


def manhattanMesafesi(test_verileri, egitim_verisi, len_testveri):
    mesafe = 0
    for i in range(len_testveri):
            mesafe += (abs(int(test_verileri[i]) - int(egitim_verisi[i])))
    return mesafe


def minkowskiMesafesi(test_verileri, egitim_verisi, len_testveri):
    p=3
    mesafe=0
    for x in range (len_testveri):
        mesafe +=pow(abs(int(test_verileri[x])-int(egitim_verisi[x])),p)
        mesafe +=pow(mesafe,(1/p))
    return mesafe



#İlk olarak parametre olarak verilen hesaplama türüne göre test verilerinin eğitim verilerine göre mesafelerine bakılır.mesafe değişkenine atılır.
#Daha sonra her test verisinin hangi eğitim verisine ne kadar uzaklıkta olduğu mesafesi ile beraber mesafeler dizisine atılır.
#mesafeler.sort ile en kısa mesafeli olan egitim verileri mesafeler dizisinde en başa gelecek şekilde sıralanır.
#En alttaki döngüde parametre olarak gelecek k değerine göre mesafeler dizisindeki ilk k eleman alınır ve en_yakinler dizisine atılır.En yakinler dizisi döndürülür.
def en_yakın_degerler(egitim_verileri, test_verisi,hesaplama_türü,k):
    mesafeler = []
    en_yakinlar = []
    test_veri_boyutu = len(test_verisi) - 1#sınıf değeri çıkarılır.

    for x in range(len(egitim_verileri)):
        mesafe = hesaplama_türü(test_verisi, egitim_verileri[x], test_veri_boyutu)
        mesafeler.append((egitim_verileri[x], mesafe))#distances dizisine eğitim verisini ve ona ait uzunlukları sırayla(x değerine bağlı) yazar.
    mesafeler.sort(key=lambda elem: elem[1])
    for x in range(k):
        en_yakinlar.append(mesafeler[x][0])
    return en_yakinlar


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
    #k=1,2,3 dışındakiler deneme amaçlı kullanılmıştır.
    k_dizisi = [1, 2, 3, 4, 5, 10, 15 , 25, 35, 45]
    hesaplama_tipi = [oklidMesafesi, manhattanMesafesi, minkowskiMesafesi]
    egitim_verileri = []
    test_verileri = []
    real_sonuclar = []
    oran = 0.75
    veriSetiYukle(r'../data/covid19.csv', oran, egitim_verileri, test_verileri)

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
        real_sonuclar.append(test_verileri[i][5])
    print("Real Sonuçlar   ", real_sonuclar)
    print('Başarı Oranı: ' + repr(basariOrani) + '%')

    #Real ve tahminler dizisindeki string degerler int veriye dönüstürülür.
    for i in range(0, len(real_sonuclar)):
        real_sonuclar[i] = int(real_sonuclar[i])
    for i in range(0, len(tahminler)):
        tahminler[i] = int(tahminler[i])
    input("\nNaive Bayes Sınıflandırma için enter tuşuna basınız")
    print("\n-----------------------------------------------------------\n      ---------  Naive Bayes Sınıflandırma  ---------     \n-----------------------------------------------------------")
    NaiveBayes(oran)
    #Programın basarı oranı hesaplatılır.Roc curve çizdirmek için alttaki kod satırlarının açılması gerekiyor.
    """
    fpr, tpr, thresholds = roc_curve(real_sonuclar, tahminler)
    plt.figure(figsize=(3, 3))
    graph(fpr, tpr, basariOrani)
    plt.show()
    """

#Kullandığım verisetine göre program %100 e yakın başarı oranı vermektedir.K değerinin yüksek verilmesi durumunda oran %90 lara düşmektedir.
#Deneme aşamasında kullandığım diğer verisetlerinde bu oran %75 ile %95 arasında değişmekteydi.
def kullanıcı_tarafi():
    # prepare data
  Pca()
  while True:
    print("\n------------------------------------------------------\n      ---------  Knn Sınıflandırma  ---------     \n------------------------------------------------------")
    print(
        "Knn sınıflandırma için projede 3 uzaklık algoritması kullanılmıştır.Bunlar ;\nÖklid Mesafesi \nManhattan Mesafesi \nMinkowski Mesafesi\n\nÖklid için p değerine '1',Manhattan için p değerine '2',Minkowski için p değerine '3' giriniz...")
    p=int(input("P değeri giriniz"))
    if(p == 1):
        print("Hesaplamada kullanılacak mesafe türü : Öklid Mesafesi\n")
    elif(p == 2):
        print("Hesaplamada kullanılacak mesafe türü : Manhattan Mesafesi\n")
    elif (p == 2):
        print("Hesaplamada kullanılacak mesafe türü : Minkowski Mesafesi\n")
    p=p-1
    print("Hesaplama yapılırken kaç komşuya göre sınıflandırma yapmak istediğinizi(k) giriniz.")
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
    input("\nKnn Sınıflandırmayı tekrar çalıştırmak için enter tuşuna basınız")
kullanıcı_tarafi()

