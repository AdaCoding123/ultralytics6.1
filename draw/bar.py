from matplotlib import pyplot as plt
from matplotlib import font_manager

data = {
    " YOLOv8 ":[75.7, 99.5, 78.4, 89.5, 93.6, 0, 99.5, 99.3],
    " Improved YOLOv8 ": [98.2, 80.2, 92.7, 99.5, 99.4, 85.7, 91.7, 98.5],
}
a=['star_crack', 'short_circuit', 'crack', 'thick_line', 'finger', 'vertical_dislocation', 'black_core','horizontal_dislocation']

#设置图形大小
plt.figure(figsize=(15,12),dpi = 80)

#数据
b_1 = [75.7, 99.5, 78.4, 89.5, 93.6, 0, 99.5, 99.3]
b_2 = [98.2, 80.2, 92.7, 99.5, 99.4, 85.7, 91.7, 98.5]

height = 0.2
a1 = list(range(len(a)))
a2 = [i+height for i in a1]#坐标轴偏移
a3 = [i+height*2 for i in a1]

#绘图
bar1=plt.barh(a1,b_1,height= height,label = "YOLOv8",color = "#969696")

plt.bar_label(bar1, label_type='edge')
    
bar2=plt.barh(a2,b_2,height= height,label = "Improved YOLOv8",color = "#FFC125")
plt.bar_label(bar1, label_type='edge')

#绘制网格
plt.grid(alpha = 0.4)

#y轴坐标刻度标识
plt.yticks(a2,a,fontsize = 14)

#添加图例
plt.legend()

#添加横纵坐标，标题
plt.xlabel("mAP@0.5",fontsize = 16)
# plt.ylabel("电影名称",fontsize = 16)
# plt.title("1,2,3号电影实时票房统计图",fontsize = 24)

#显示图形
plt.show()
