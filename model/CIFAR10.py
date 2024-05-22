from keras.optimizers import Adamax
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import save_model, load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization


"""
Giriş katmanı: 32x32x3 boyutlarında bir resim (RGB kanalları varsayılarak).
32 filtre içeren, her biri (3, 3) boyutlarında olan ReLU aktivasyonu kullanan evrişim katmanı.
Maksimum Havuzlama katmanı, boyutları azaltmak için (2, 2) havuzlama boyutu kullanır.
64 filtre içeren, her biri (3, 3) boyutlarında olan ReLU aktivasyonu kullanan bir evrişim katmanı.
Maksimum Havuzlama katmanı, boyutları azaltmak için (2, 2) havuzlama boyutu kullanır.
2D özellik haritalarını 1D vektöre dönüştürmek için Düzleştirme katmanı.
512 birim içeren ve ReLU aktivasyonu kullanan Tam Bağlantılı (Dense) katman.
Aşırı uyumunu önlemek için dropout oranı 0.5 olan Dropout katmanı.
Çoklu sınıf sınıflandırması için 10 birim içeren çıkış katmanı ve softmax aktivasyonu.
Son olarak, model, belirli bir öğrenme oranı, kategorik çapraz entropi kaybı ve doğruluk metriği ile Adamax optimizer kullanılarak derlenir.
"""

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Bu işlem, görüntü verilerini normalize etmek içindir.
#Görüntü pikselleri genellikle 0 ile 255 arasında değerlere sahiptir.
#Bu değerlerin 0 ile 1 arasına ölçeklendirilmesi, modelin daha iyi performans göstermesine yardımcı olabilir.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
"""
Evrişim Katmanı (Conv2D):

Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)) ifadesi, 32 adet filtre kullanılarak, her biri 3x3 boyutunda olan evrişim filtrelerini uygulayan bir evrişim katmanını tanımlar.
Bu katman, giriş verisi üzerinde evrişim işlemini uygular. activation='relu' ifadesi, ReLU (Rectified Linear Unit) aktivasyon fonksiyonunu kullanmasını belirtir. ReLU, negatif değerleri sıfır yaparak pozitif değerleri geçirir.
input_shape=(32, 32, 3) ifadesi, giriş verisinin boyutunu belirtir. Bu durumda 32x32 piksel boyutunda ve 3 kanallı (RGB renk kanalları) bir giriş beklenmektedir.
MaxPooling Katmanı (MaxPooling2D):

MaxPooling2D(pool_size=(2, 2)) ifadesi, maksimum havuzlama (max pooling) işlemini uygulayan bir maksimum havuzlama katmanını tanımlar.
Maksimum havuzlama, her bir bölgenin içinden en büyük değeri seçerek boyutu küçültür. pool_size=(2, 2) ifadesi, her bir havuzlama bölgesinin boyutunu 2x2 olarak belirtir.
"""
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(BatchNormalization()) # Batch normalization, eğitimi daha hızlı ve daha istikrarlı hale getirerek öğrenmeyi artırabilir. Her evrişim veya tam bağlantılı katmanın hemen ardından ekleyebilirsiniz.

model.add(Flatten()) # 32x32x32 vektörü bir boyutlu vektöre dönüştüren düzleştirme katmanı.
model.add(Dense(512, activation='relu')) #512 nöronluk bağlantı katmanı
model.add(BatchNormalization())

model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax')) #10 nöronluk çıkış katmanı

optimizer = Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

with tf.device('/GPU:0'):
    model_train = model.fit(x_train, y_train, epochs=60, batch_size=32, validation_data=(x_val, y_val))

model_json = model.to_json()
open('CNN_three_layer_fully_connected.json', 'w').write(model_json)
model.save_weights('CNN_three_fully_connected.h5', overwrite=True)

print(model_train.history.keys())

# Eğitim geçmişini çizme
plt.plot(model_train.history['accuracy'], label='Training Accuracy')
plt.plot(model_train.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.plot(model_train.history['loss'], label='Training Loss')
plt.plot(model_train.history['val_loss'], label='Validation Loss')
plt.ylabel('Loss')
plt.ylabel('Accuracy')
print("Model Summary =>", model.summary())
plt.legend()
plt.show()

