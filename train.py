# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import cv2
# from glob import glob
# from tqdm import tqdm
# import re
# from Config import Config
# import random
# import numpy as np
# from modle import FaceNetmModel
# from torch.nn import functional as F
#
# # 从test.json文件中读取配置
# config = Config.from_json_file("test.json")
#
#
# def read_data():
#     # 获取所有图片路径
#     paths = glob("./test/*")
#     paths_dict = dict()
#     # 将图片路径按照人名进行分类
#     for path in paths:
#         man_num = re.findall(r"(\d+)-\d+\.", path)[0]
#         if man_num not in paths_dict:
#             paths_dict[man_num] = [path]
#         else:
#             paths_dict[man_num].append(path)
#     keys = list(paths_dict.keys())
#     # 设置类别数量
#     Config.class_nums = len(keys)
#     new_paths = []
#     # 遍历所有类别
#     for i in range(Config.class_nums):
#         paths_ys = paths_dict[keys[i]]
#         paths_yen = len(paths_ys)
#         # 遍历每个类别的图片
#         for image_path in paths_ys:
#             new_path = [image_path]
#             # 随机选择一张图片作为正样本
#             rand_num = random.randint(0, paths_yen - 1)
#             new_path.append(paths_ys[rand_num])
#
#             # 随机选择一个不同类别的图片作为负样本
#             rand_num_man = random.randint(0, Config.class_nums - 1)
#             if rand_num_man == i:
#                 try:
#                     rand_num_man += 1
#                     n_keys = keys[rand_num_man]
#                 except:
#                     rand_num_man -= 1
#                     n_keys = keys[rand_num_man]
#             else:
#                 n_keys = keys[rand_num_man]
#
#             n_path = paths_dict[n_keys]
#             n_num = random.randint(0, len(n_path) - 1)
#             new_path.append(n_path[n_num])
#
#             new_paths.append(new_path)
#     # print(new_paths)
#     return new_paths
#
#
# class FaceData(DataLoader):
#     def __init__(self, paths):
#         self.paths = paths
#
#     def read_image(self, path):
#         # 读取图片并进行预处理
#         image = cv2.imread(path)
#         image = cv2.resize(image, (config.image_size, config.image_size)) / 127
#         image = np.transpose(image, (2, 0, 1))
#         return image
#
#     def __getitem__(self, index):
#         # 获取三元组图片路径
#         a_path, p_path, n_path = self.paths[index]
#         # 读取图片并进行预处理
#         a_img, p_img, n_img = self.read_image(a_path), self.read_image(p_path), self.read_image(n_path)
#         # 获取图片对应的类别
#         s_1 = int(re.findall(r"(\d+)-\d+\.", a_path)[0]) - 100
#         n_1 = int(re.findall(r"(\d+)-\d+\.", n_path)[0]) - 100
#         return np.float32(a_img), np.float32(p_img), np.float32(n_img), np.int64(s_1), np.int64(n_1)
#
#     def __len__(self):
#         return len(self.paths)
#
#
# class TripletLoss(nn.Module):
#     def __init__(self, alpha):
#         super().__init__()
#         self.alpha = alpha
#         self.pairwise_distance = nn.PairwiseDistance()
#
#     def forward(self, anchor, positive, negative):  # 计算三元组损失
#         pos_dist = self.pairwise_distance(anchor, positive)
#         neg_dist = self.pairwise_distance(anchor, negative)
#         loss = torch.clamp(pos_dist - neg_dist + self.alpha, min=0.0)
#         return loss
#
#
# def train():
#     # 读取数据
#     paths = read_data()
#
#     train_data = FaceData(paths)
#     # 将数据加载到DataLoader中
#     train_data = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.batch_size)
#
#     # 初始化模型
#     model = FaceNetmModel(config.emd_size, config.class_nums)
#     model.train()
#
#     # 初始化优化器
#     optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
#     # 初始化三元组损失函数
#     loss_t_fc = TripletLoss(alpha=config.alpha)
#     # 初始化交叉熵损失函数
#     loss_c_fc = nn.CrossEntropyLoss()
#     # 获取数据集大小
#     nb = len(train_data)
#
#     # 遍历所有epoch
#     for epoch in range(1, config.epochs + 1):
#         pbar = tqdm(train_data, total=nb)
#         # 遍历每个batch
#         for step, (a_x, p_x, n_x, s_y, n_y) in enumerate(pbar):
#
#             # 获取三元组图片特征
#             a_out, p_out, n_out = model(a_x), model(p_x), model(n_x)
#             # 计算三元组损失
#             s_d = F.pairwise_distance(a_out, p_out)
#             n_d = F.pairwise_distance(a_out, n_out)
#             thing = (n_d - s_d < config.alpha).flatten()
#             mask = np.where(thing.numpy() == 1)
#             if not len(mask):
#                 continue
#             a_out, p_out, n_out = a_out[mask], p_out[mask], n_out[mask]
#             loss_t = torch.mean(loss_t_fc(a_out, p_out, n_out))
#
#             # 获取三元组图片
#             a_x, p_x, n_x = a_x[mask], p_x[mask], n_x[mask]
#             # 获取三元组图片对应的类别
#             s_y, n_y = s_y[mask], n_y[mask]
#
#             # 将三元组图片拼接在一起
#             input_x = torch.cat([a_x, p_x, n_x])
#             # 将三元组图片对应的类别拼接在一起
#             output_y = torch.cat([s_y, s_y, n_y])
#             # 获取模型输出
#             out = model.forward_class(input_x)
#             # 计算交叉熵损失
#             loss_c = torch.mean(loss_c_fc(out, output_y))
#
#             # 计算总损失
#             loss = loss_t + loss_c
#
#             # 反向传播
#             loss.backward()
#             # 更新参数
#             optimizer.step()
#             # 清空梯度
#             optimizer.zero_grad()
#
#             # 打印损失
#             s = ("train ===> epoch:{} ---- step:{} ---- loss_t:{} ---- loss_c:{}").format(epoch, step, loss_t, loss_c)
#             pbar.set_description(s)
#             # 每隔5个epoch保存模型
#             if epoch % 5 == 0:
#                 torch.save(model.state_dict(), "facenet.pth")
#
#
# if __name__ == "__main__":
#     train()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
from glob import glob
from tqdm import tqdm
import re
from Config import Config
import random
import numpy as np
from modle import FaceNetmModel
from torch.nn import functional as F

# 从test.json文件中读取配置
config = Config.from_json_file("test.json")


def read_data():
    # 获取所有图片路径
    paths = glob("./test/*")
    paths_dict = dict()
    # 将图片路径按照人名进行分类
    for path in paths:
        man_num = re.findall(r"(\d+)-\d+\.", path)[0]
        if man_num not in paths_dict:
            paths_dict[man_num] = [path]
        else:
            paths_dict[man_num].append(path)
    keys = list(paths_dict.keys())
    # 设置类别数量
    Config.class_nums = len(keys)
    new_paths = []
    # 遍历所有类别
    for i in range(Config.class_nums):
        paths_ys = paths_dict[keys[i]]
        paths_yen = len(paths_ys)
        # 遍历每个类别的图片
        for image_path in paths_ys:
            new_path = [image_path]
            # 随机选择一张图片作为正样本
            rand_num = random.randint(0, paths_yen - 1)
            new_path.append(paths_ys[rand_num])

            # 随机选择一个不同类别的图片作为负样本
            rand_num_man = random.randint(0, Config.class_nums - 1)
            if rand_num_man == i:
                try:
                    rand_num_man += 1
                    n_keys = keys[rand_num_man]
                except:
                    rand_num_man -= 1
                    n_keys = keys[rand_num_man]
            else:
                n_keys = keys[rand_num_man]

            n_path = paths_dict[n_keys]
            n_num = random.randint(0, len(n_path) - 1)
            new_path.append(n_path[n_num])

            new_paths.append(new_path)
    # print(new_paths)
    return new_paths


class FaceData(DataLoader):
    def __init__(self, paths):
        self.paths = paths

    def read_image(self, path):
        # 读取图片并进行预处理
        image = cv2.imread(path)
        image = cv2.resize(image, (config.image_size, config.image_size)) / 127
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, index):
        # 获取三元组图片路径
        a_path, p_path, n_path = self.paths[index]
        # 读取图片并进行预处理
        a_img, p_img, n_img = self.read_image(a_path), self.read_image(p_path), self.read_image(n_path)
        # 获取图片对应的类别
        s_1 = int(re.findall(r"(\d+)-\d+\.", a_path)[0]) - 100
        n_1 = int(re.findall(r"(\d+)-\d+\.", n_path)[0]) - 100
        return np.float32(a_img), np.float32(p_img), np.float32(n_img), np.int64(s_1), np.int64(n_1)

    def __len__(self):
        return len(self.paths)


class TripletLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.pairwise_distance = nn.PairwiseDistance()

    def forward(self, anchor, positive, negative):  # 计算三元组损失
        pos_dist = self.pairwise_distance(anchor, positive)
        neg_dist = self.pairwise_distance(anchor, negative)
        loss = torch.clamp(pos_dist - neg_dist + self.alpha, min=0.0)
        return loss


def train():
    # 读取数据
    paths = read_data()

    train_data = FaceData(paths)
    # 将数据加载到DataLoader中
    train_data = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.batch_size)

    # 初始化模型
    model = FaceNetmModel(config.emd_size, config.class_nums)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # 初始化三元组损失函数
    loss_t_fc = TripletLoss(alpha=config.alpha)
    # 初始化交叉熵损失函数
    loss_c_fc = nn.CrossEntropyLoss()
    # 获取数据集大小
    nb = len(train_data)

    # 遍历所有epoch
    for epoch in range(1, 20):
        pbar = tqdm(train_data, total=nb)
        # 遍历每个batch
        for step, (a_x, p_x, n_x, s_y, n_y) in enumerate(pbar):
            a_x, p_x, n_x, s_y, n_y = a_x.to(device), p_x.to(device), n_x.to(device), s_y.to(device), n_y.to(device)
            # 获取三元组图片特征
            a_out, p_out, n_out = model(a_x), model(p_x), model(n_x)
            # 计算三元组损失
            s_d = F.pairwise_distance(a_out, p_out)
            n_d = F.pairwise_distance(a_out, n_out)
            thing = (n_d - s_d < config.alpha).flatten()
            mask = np.where(thing.cpu().numpy() == 1)
            if not len(mask):
                continue
            a_out, p_out, n_out = a_out[mask], p_out[mask], n_out[mask]
            loss_t = torch.mean(loss_t_fc(a_out, p_out, n_out))

            # 获取三元组图片
            a_x, p_x, n_x = a_x[mask], p_x[mask], n_x[mask]
            # 获取三元组图片对应的类别
            s_y, n_y = s_y[mask], n_y[mask]

            # 将三元组图片拼接在一起
            input_x = torch.cat([a_x, p_x, n_x])
            # 将三元组图片对应的类别拼接在一起
            output_y = torch.cat([s_y, s_y, n_y])
            # 获取模型输出
            out = model.forward_class(input_x)
            # 计算交叉熵损失
            loss_c = torch.mean(loss_c_fc(out, output_y))

            # 计算总损失
            loss = loss_t + loss_c

            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 清空梯度
            optimizer.zero_grad()

            # 打印损失
            s = ("train ===> epoch:{} ---- step:{} ---- loss_t:{} ---- loss_c:{}").format(epoch, step, loss_t, loss_c)
            pbar.set_description(s)
            # 每隔2个epoch保存模型
            if epoch % 2 == 0:
                torch.save(model.state_dict(), "facenet.pth")


if __name__ == "__main__":
    train()
