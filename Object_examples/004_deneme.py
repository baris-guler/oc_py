
import sys
sys.path.append('../')  # 'class_func.py' dosyası üst klasörde olduğu için onu path'e eklendi

import class_func as cf # 'class_func.py' dosyasını import edildi
import numpy as np # numpy kütüphanesini import edildi
from matplotlib import pyplot as plt
epochs = np.linspace(-30000, 55000, 60000) # Modelin uygulanacağı epoch sayı dizisini oluşturuldu
func = cf.OC_model(epochs=epochs) # Toplam bir OC model oluşturmak için oc_model objesi oluşturuldu. Bu modele istenen evre değerlerini vermek için 'epoch' parametresine x değeri verildi

lin = cf.Lin() # OC model nesnesine eklenmek için lin model bileşeni oluşturuldu
lin.dP = 0.370564
lin.dT = 39954.025
quad = cf.Quad() # OC model nesnesine eklenmek için quad model bileşeni oluşturuldu
quad.Q = 2.01647235790475E-10
lite = cf.LiTE() # OC model nesnesine eklenmek için LiTE model bileşeni oluşturuldu
lite.amp = 0.0402322950352529
lite.e = 0.542897802
lite.P_LiTE = 55298.9270567742
lite.T_LiTE = 40414.1933167451
lite.omega = 73.2665401924


func.add_model_component(lin) # OC nesnesine lin bileşeni eklendi
func.add_model_component(quad) # OC nesnesine lin bileşeni eklendi
func.add_model_component(lite) # OC nesnesine lin bileşeni eklendi
data = cf.OC_data(data_file="../xy_boo.xlsx")
fits = cf.fit(func, data)
result = fits.fit_dynamic_model()