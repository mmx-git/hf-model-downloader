# 打包成 Windows exe 的步骤

## 1. 准备环境（在你的 Windows 电脑上操作）

安装 Python（如果还没有）：https://www.python.org/downloads/windows/
安装时记得勾选 "Add python.exe to PATH"。

打开命令提示符（cmd 或 PowerShell），进入 app.py 所在文件夹，执行：

```
pip install -r requirements.txt
```

## 2. 下载 aria2c.exe（必需，这是实际负责下载的组件）

访问 https://github.com/aria2/aria2/releases

下载最新的 Windows 版本，文件名类似 `aria2-1.37.0-win-64bit-build1.zip`，解压后把里面的
`aria2c.exe` 复制到跟 `app.py` 同一个文件夹里。

## 3. 打包成单个 exe 文件

在同一个文件夹下执行：

```
pyinstaller --onefile --name HF-Model-Downloader --add-data "aria2c.exe;." app.py
```

打包完成后，exe 文件会出现在 `dist` 文件夹里：`dist\HF-Model-Downloader.exe`

## 4. 使用

双击 `dist\HF-Model-Downloader.exe`，稍等几秒会自动打开浏览器访问工具页面。

如果杀毒软件报警（PyInstaller 打包的 exe 有时会被误报），选择"允许运行"即可，
这是 PyInstaller 打包程序的通病，不是这个工具本身有问题。

## 5. 常见问题

**Q: 双击后黑窗口一闪而过，没打开浏览器？**
A: 命令行方式运行看报错：在 cmd 里 cd 到 dist 文件夹，直接输入 `HF-Model-Downloader.exe` 回车，
   看看具体报什么错，通常是 aria2c.exe 没有放对位置。

**Q: 想要没有黑色控制台窗口，双击直接静默运行？**
A: 打包命令加上 `--windowed` 参数：
   ```
   pyinstaller --onefile --windowed --name HF-Model-Downloader --add-data "aria2c.exe;." app.py
   ```
   注意：加了 --windowed 之后，如果程序报错你会看不到任何提示，调试的时候建议先不加这个参数。

**Q: 保存目录选哪里？**
A: Windows 版本默认保存目录是当前用户的 `下载\HF-Models` 文件夹，也可以在网页里点"浏览..."
   自己选别的磁盘和文件夹。
