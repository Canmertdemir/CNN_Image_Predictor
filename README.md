# CNN_Image_Predictor

# CNN Modeli Kullanımı

Bu proje, CIFAR-10 veri kümesi üzerinde bir CNN modelinin eğitimini ve sonrasında bu modelin kullanımını içerir.

## Kullanım

1. **Gereksinimler**

   - Python 3.x
   - Keras
   - TensorFlow
   - NumPy
   - Matplotlib
   - Jupyter Notebook (isteğe bağlı, eğitim ve tahminlerinizi görselleştirmek için kullanılabilir)

2. Kurulum

   Gereksinimleri yüklemek için aşağıdaki komutları kullanın:

   ```bash
   pip install numpy matplotlib tensorflow keras jupyter
   ```

3. Veri Kümesi

   CIFAR-10 veri kümesi, derin öğrenme modeli eğitimi ve testi için kullanılacaktır. Veri kümesi otomatik olarak yüklenir.

4. Eğitim

   Modeli eğitmek için `CNN_model_egitim.ipynb` Jupyter Notebook dosyasını kullanın.

   ```bash
   jupyter notebook CNN_model_egitim.ipynb
   ```

   Notebook içindeki talimatları takip ederek modeli eğitebilirsiniz.


Aşağıda, FastAPI kullanarak bir görüntü sınıflandırma uygulaması için bir "readme" şablonu bulunmaktadır:

---

# Görüntü Sınıflandırma Uygulaması

Bu proje, FastAPI kullanarak bir görüntü sınıflandırma servisi sunar. Kullanıcılar, uygulamaya bir görüntü yükleyerek, bu görüntünün hangi sınıfa ait olduğunu tahmin edebilirler.

## Kurulum

1. **Gereksinimler**

   - Python 3.x
   - FastAPI
   - Uvicorn
   - Python-Multipart
   - TensorFlow
   - OpenCV
   - NumPy

   Gereksinimleri yüklemek için aşağıdaki komutları kullanın:

   ```bash
   pip install fastapi uvicorn python-multipart tensorflow opencv-python-headless numpy
   ```

2. **Model ve Servis Hazırlığı**

   - Görüntü sınıflandırma modeli ve ağırlıkları (`CNN_three_layer_fully_connected.json` ve `CNN_three_fully_connected.h5`) bulunmalıdır. Bu dosyaların, bu proje dizininde olduğundan emin olun.
   - Servis, görüntüyü işlemek ve tahminler yapmak için OpenCV ve NumPy kütüphanelerini kullanır. Gerekirse bu kütüphaneleri yükleyin.

3. **Uygulamayı Başlatma**

   Uygulamayı başlatmak için aşağıdaki komutu kullanın:

   ```bash
   uvicorn app:app --reload
   ```

   Bu komut, `app.py` dosyasındaki FastAPI uygulamasını başlatır ve değişiklikler otomatik olarak algılanarak yeniden yüklenir.

4. **Kullanım**

   - Uygulama başlatıldıktan sonra, bir web tarayıcısında `http://localhost:8000` adresine gidin.
   - Sayfada, bir görüntü dosyası seçin ve "Yükle" düğmesine basın.
   - Servis, yüklenen görüntüyü alacak ve sınıflandıracak ve sonucu ekranda gösterecektir.

## Notlar
- Bu uygulama, önceden eğitilmiş bir görüntü sınıflandırma modelini kullanır. Modeli eğitmek ve yeni verilere göre güncellemek istiyorsanız, ayrıntılı talimatlar için model belgelerine bakın.
- FastAPI hizmetine bir görüntü yüklemek için basit bir HTML formu bulunmaktadır. Bu form, kullanıcıların bir görüntü seçmelerine ve sunucuya yüklemelerine olanak tanır. Yüklenen görüntü, FastAPI hizmeti tarafından tahmin edilir ve sonuç JSON olarak sunulur. Ayrıca, CIFAR-10 veri seti ve CNN modeli hakkında bilgilendirici bir bölüm de içerir.
  
Referanslar
github.com/ABI-Virtual-Brain-Project/CNN-TaskLisansa tabi (Apache - 2.0)
github.com/liyongqingupc/ACLN-WindFieldCorrection
github.com/AlexTorres10/TCC
github.com/Mehwish4593/Mehwish4593
stackoverflow.com/questions/54207049/keras-cnn-validation-accuracy-stuck-at-70-training-accuracy-reaching-100
github.com/Chetank190/DogCatsClassifier_BigDataProject
github.com/Catalina-13/City-Congestion-Simulator
github.com/raja21068/Android_Malware_DetectionLisansa tabi (Apache - 2.0)
https://baraaalbourghli.medium.com/deploy-your-keras-or-tensorflow-machine-learning-model-to-aws-using-amazon-sagemaker-how-to-2d88a6e779cc
