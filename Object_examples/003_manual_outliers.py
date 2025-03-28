import sys
sys.path.append('../')
import class_func as cf

data2 = cf.OC_data(data_file="003_xy_boo - mintime.xlsx")
outliers = data2.manual_outliers()
data2.remove_outliers(outliers)

data2.plot_OC()