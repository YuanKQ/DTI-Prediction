# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: linechart.py
@time: 2019/3/24 12:00
@description:
"""

from matplotlib import pyplot
import matplotlib.pyplot as plt

# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

xticks_rotation = 25

def plot_semantic_line_chart(y_values, x_label, y_label, x_ticks, y_ticks):
    x = range(len(x_ticks))
    plt.plot(x, y_values[0], "--", label="pharmacology attention")
    plt.plot(x, y_values[1], "-.", label="class attention")
    plt.plot(x, y_values[2], ":", label="co-attention")
    plt.plot(x, y_values[3], "rs-", label="multi-view attention")
    plt.legend(loc="upper right")  # 图例
    plt.margins(0)
    plt.xlabel(x_label)  # X轴标签
    plt.ylabel(y_label)  # Y轴标签
    plt.xticks(x, x_ticks, rotation=xticks_rotation)  # 设置横坐标数值
    pyplot.yticks(y_ticks)  # 设置纵坐标数值
    plt.show()


def plot_class_line_chart(y_values, x_label, y_label, x_ticks, y_ticks):
    x = range(len(x_ticks))
    plt.plot(x, y_values[0], "--", label="pharmacology attention")
    plt.plot(x, y_values[1], "-.", label="semantic attention")
    plt.plot(x, y_values[2], ":", label="co-attention")
    plt.plot(x, y_values[3], "rs-", label="multi-view attention")
    plt.legend(loc="upper right")  # 图例
    plt.margins(0)
    plt.xlabel(x_label)  # X轴标签
    plt.ylabel(y_label)  # Y轴标签
    plt.xticks(x, x_ticks, rotation=xticks_rotation)  # 设置横坐标数值
    pyplot.yticks(y_ticks)  # 设置纵坐标数值
    # plt.title('Scatter plot pythonspot.com', y=-0.17)
    plt.show()


y_label = "Attention Weight"#"注意力权重"
#cefotaxime
y_semantic = [[0.56375684, 0.2273922, 0.04835121, 0.0733506, 0.01178294, 0.01496016, 0.02562782, 0.00650471],
               [0.21975984, 0.16806135, 0.16233777, 0.26201263, 0.03130473, 0.03130473, 0.03130473, 0.03130473],
               [0.16845249, 0.16145247, 0.16144093, 0.15928379, 0.05939505, 0.05939505, 0.05939505, 0.05939505],
               [0.18279171, 0.12400032, 0.1165156, 0.10308027, 0.07871714, 0.07896763, 0.07981454, 0.07830274]]
y_class = [[0.24429433, 0.01869775, 0.01081546, 0.00551313, 0.00227775, 0.0016846, 0.05437766, 0.30140708],
          [0.3237772, 0.22907827, 0.11413021, 0.03281201, 0.0105448, 0.02076072, 0.0672242, 0.0672242],
          [0.13384161, 0.1284161, 0.13384161, 0.13384161, 0.04161, 0.0384161, 0.04923758, 0.04923758],
          [0.15717877, 0.12684215, 0.09449266, 0.09065213, 0.08447022, 0.08028698, 0.07653719, 0.07078636]]
x_semantic_label = "(a) Multi-view attention on Hierarchical Nodes: about the Semantic Feature of Cefotaxime " #"(a) 多元特征融合算法对头孢噻肟的语义表示中的各路径节点的影响"
x_class_label = "(b) Multi-view attention on Hierarchical Nodes: about the Class Feature of Cefotaxime " #"(b)多元特征融合算法对头孢噻肟的分类层次结构编码中的各路径节点的影响"
x_ticks = ["Cefotaxime", "Cephalosporin\n3'-esters", "Cephalosporins", "Cephems", "Beta Lactams", "Lactams", "Organoheterocyclic\nCompounds", "Organic Compounds"]#["头孢噻肟", "三代头孢", "头孢菌素", "头孢烯", "Beta内酰胺", "内酰胺", "有机杂环化合物", "有机化合物"]
y_semantic_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]   #[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
y_class_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]   # [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
plot_semantic_line_chart(y_semantic, x_semantic_label, y_label, x_ticks, y_semantic_ticks)
plot_class_line_chart(y_class, x_class_label, y_label, x_ticks, y_class_ticks)

#ceftriaxone
y_semantic = [[0.35044444, 0.11775191, 0.04727394, 0.03232001, 0.07901495, 0.06401032, 0.10800061, 0.04443964],
              [0.35093959, 0.22301335, 0.13412667, 0.14198298, 0.02498957, 0.02498957, 0.02498957, 0.02498957],
              [0.1610462, 0.16073849, 0.16104023, 0.1612305, 0.0593241, 0.0593241, 0.0593241, 0.0593241],
              [0.208201159, 0.17080755, 0.11905803, 0.10154967, 0.10085065, 0.08489675, 0.08363242, 0.08739355]]
y_class = [[0.12015176, 0.03221731, 0.05801116, 0.05569391, 0.08589816, 0.13215874, 0.0387904, 0.21631826],
           [0.28437437, 0.17280425, 0.02727437, 0.01085692, 0.08411501, 0.08411501, 0.08411501, 0.08411501],
           [0.16053025, 0.16626219, 0.16120832, 0.16098469, 0.05933576, 0.05933576, 0.05933576, 0.05933576],
           [0.13249658, 0.110615304, 0.09416909, 0.09240061, 0.09256833, 0.08695119, 0.08830876, 0.080546373]]
x_semantic_label = "(c) Multi-view attention on Hierarchical Nodes: about the Semantic Feature of Ceftriaxone "# "(c) 多元特征融合算法对头孢曲松的语义表示中的各路径节点的影响"
x_class_label = "(d) Multi-view attention on Hierarchical Nodes: about the Class Feature of Ceftriaxone "#"(d)多元特征融合算法对头孢曲松的分类层次结构编码中的各路径节点的影响"
x_ticks = ["Ceftriaxone", "Cephalosporin\n3'-esters", "Cephalosporins", "Cephems", "Beta Lactams", "Lactams", "Organoheterocyclic\nCompounds", "Organic\nCompounds"] #["头孢曲松", "三代头孢", "头孢菌素", "头孢烯", "Beta内酰胺", "内酰胺", "有机杂环化合物", "有机化合物"]
y_semantic_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]   # [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
y_class_ticks =[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]   #[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
plot_semantic_line_chart(y_semantic, x_semantic_label, y_label, x_ticks, y_semantic_ticks)
plot_class_line_chart(y_class, x_class_label, y_label, x_ticks, y_class_ticks)

#meropenem
y_semantic = [[0.36219199, 0.40794158, 0.04317219, 0.09448496, 0.01048007, 0.01881377, 0.03013998],
              [0.72669242, 0.09276443, 0.05315886, 0.01427348, 0.0188518, 0.0188518, 0.0188518],
              [0.17111385, 0.16092381, 0.16051466, 0.06083812, 0.05943493, 0.05943493, 0.05943493],
              [0.23873871, 0.13255547, 0.08843052, 0.095653, 0.07474918, 0.07537472, 0.07623328]]

y_class = [[0.03172922, 0.03969637, 0.11551017, 0.34346198, 0.12586285, 0.05834027, 0.0181773],
           [0.0427414, 0.06319577, 0.10125594, 0.05849087, 0.122386, 0.122386, 0.122386],
           [0.18505244, 0.18066226, 0.15642413, 0.06826588, 0.06826588, 0.06826588, 0.06826588],
           [0.13978046, 0.119811007, 0.10731043, 0.12024448, 0.10139769, 0.0947771, 0.09104599]]

x_semantic_label = "(e) Multi-view attention on Hierarchical Nodes: about the Semantic Feature of Meropenem" #"(e) 多元特征融合算法对美罗培南的语义表示中的各路径节点的影响"
x_class_label = "(f) Multi-view attention on Hierarchical Nodes: about the Class Feature of Meropenem" #"(f)多元特征融合算法对美罗培南的分类层次结构编码中的各路径节点的影响"
x_ticks = ["Meropenem", "Thienamycins", "Carbapenems", "Beta Lactams", "Lactams", "Organoheterocyclic\nCompounds", "Organic\nCompounds"] # ["美罗培南", "噻烯霉素", "碳青霉烯类", "Beta内酰胺", "内酰胺", "有机杂环化合物", "有机化合物"]
y_semantic_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
y_class_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]   # [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
plot_semantic_line_chart(y_semantic, x_semantic_label, y_label, x_ticks, y_semantic_ticks)
plot_class_line_chart(y_class, x_class_label, y_label, x_ticks, y_class_ticks)


