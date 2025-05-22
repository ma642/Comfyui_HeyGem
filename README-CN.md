[中文](README-CN.md) | [English](README.md) 

# ComfyUI 的 HeyGem 数字人节点

目前 (2025.05.22) 最好的开源数字人, 没有之一. 基本可生成全身, 动态, 任意分辨率数字人.

![image](https://github.com/billwuhao/Comfyui_HeyGem/blob/main/images/2025-05-22_22-41-52.png)

## 📣 更新

[2025-05-22]⚒️: 发布 v1.0.0.

## 节点安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/Comfyui_HeyGem.git
```

## WSL 和 Docker 安装

- Windows (以 X64 为例):

1, 安装 Windows 的 Linux 子系统: https://github.com/microsoft/WSL/releases (`wsl.2.5.7.0.x64.msi`). 已经安装的, 管理员权限执行 `wsl --update` 更新.

2, 安装 Docker: https://www.docker.com/ (下载 `AMD64` 版本). 安装完成后, 启动它:

![](https://github.com/duixcom/Duix.Heygem/raw/main/README.assets/8.png)
![](https://github.com/duixcom/Duix.Heygem/raw/main/README.assets/13.png)
![](https://github.com/duixcom/Duix.Heygem/raw/main/README.assets/3.png)

镜像默认下载到 C 盘 (大概需要 14g 空间), 可以启动后在设置里修改为其他盘:

![](https://github.com/duixcom/Duix.Heygem/raw/main/README.assets/7.png)

OK! 准备就绪, 第一次运行节点, 需要下载镜像, 大概 30 分钟左右, 看网速. 不复杂, 就这么简单.

- Linux (以 Ubuntu 为例):

1, 安装 Docker: 

## 鸣谢

- [Duix.Heygem](https://github.com/duixcom/Duix.Heygem) 
- https://github.com/duixcom/Duix.Heygem/blob/main/LICENSE