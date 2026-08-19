# -*- coding: utf-8 -*-
"""
飞牛NAS - HuggingFace 模型批量下载工具
================================

功能：
  1. 输入 HuggingFace 仓库 ID（或直接粘贴仓库网址），自动拉取该仓库全部文件列表
  2. 勾选需要的文件
  3. 提交给本工具自己管理的 aria2c 后台进程批量下载（独立于 Trim，不依赖它的 RPC 密钥）
  4. 页面上可以看到 aria2 当前的下载进度

使用前准备：
  1. 确认已安装依赖：
       pip3 install flask requests --break-system-packages
  2. 确认系统已装 aria2：
       apt install aria2 -y
     （之前调试时已经装过，一般不用再装）
  3. 不需要手动找密钥 —— 本工具首次启动时会自动生成一个专属密钥，
     并且自己拉起一个独立的 aria2c RPC 守护进程（监听 127.0.0.1:6801），
     跟 Trim 自己的下载任务、RPC 完全独立，互不干扰。

运行方式：
  python3 app.py
  然后浏览器打开 http://<NAS的IP>:5678

后台运行（推荐，配合 systemd）：
  systemctl restart hf-downloader
"""

import atexit
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time

import requests
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# ========== 默认配置（也可以在网页"设置"里覆盖，会写入 config.json） ==========
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
ARIA2_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aria2-daemon.log")
ARIA2_PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aria2-daemon.pid")

import platform


def _default_save_dir():
    """跨平台默认保存目录：Windows 用用户的下载文件夹，Linux/NAS 用 /vol1"""
    if platform.system() == "Windows":
        return os.path.join(os.path.expanduser("~"), "Downloads", "HF-Models")
    return "/vol1/1000/download"


DEFAULT_CONFIG = {
    "rpc_url": "http://127.0.0.1:6801/jsonrpc",  # 本工具自建的独立 aria2 实例，跟 Trim 的 6800 分开
    "rpc_secret": "",  # 首次启动自动生成，不用手动填
    "save_dir": _default_save_dir(),
    "connections_per_file": 16,  # aria2 单文件最大连接数，1-16
    "hf_endpoint": "https://huggingface.co",  # 留空或改成镜像站地址（如 https://hf-mirror.com）
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            merged = dict(DEFAULT_CONFIG)
            merged.update(cfg)
            return merged
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def ensure_secret(cfg):
    """如果还没有密钥，自动生成一个随机密钥并写入配置文件"""
    if not cfg.get("rpc_secret"):
        cfg["rpc_secret"] = secrets.token_hex(16)
        save_config(cfg)
    return cfg


# ========== HuggingFace API ==========

def normalize_repo_id(raw: str) -> str:
    """
    自动从各种输入格式里提取出 repo_id：
    - 纯 repo_id: "user/model"
    - 完整链接（官网或任意镜像站都行）: "https://huggingface.co/user/model/tree/main"
      "https://hf-mirror.com/user/model/tree/main"
    - markdown 链接: "[user/model](https://huggingface.co/user/model/tree/main)"
    """
    raw = raw.strip()
    # markdown 链接格式，取括号里的 URL
    md_match = re.search(r'\]\((https?://[^\)]+)\)', raw)
    if md_match:
        raw = md_match.group(1)

    # 完整 URL：不管是官网还是镜像站，只要是 http(s) 链接，
    # 都提取域名后面路径的前两段（user/model），不写死域名
    if raw.startswith("http://") or raw.startswith("https://"):
        from urllib.parse import urlparse
        parsed = urlparse(raw)
        segments = [s for s in parsed.path.split("/") if s]
        if len(segments) >= 2:
            return f"{segments[0]}/{segments[1]}"
        return raw

    # 已经是纯 repo_id 格式
    return raw


def hf_list_files(repo_id: str, revision: str = "main", endpoint: str = "https://huggingface.co"):
    """调用 HF 的 tree API 获取仓库文件列表（这个接口会带上文件大小，
    普通的 /api/models/{repo_id} 接口不带 size 字段）"""
    endpoint = (endpoint or "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/api/models/{repo_id}/tree/{revision}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    files = []
    for item in data:
        if item.get("type") != "file":
            continue
        name = item.get("path")
        # 大文件走 LFS 存储，真实大小在 lfs.size 里；普通小文件用 size 字段
        lfs = item.get("lfs") or {}
        size = lfs.get("size", item.get("size"))
        if name:
            files.append({"name": name, "size": size})
    return files


def hf_resolve_url(repo_id: str, filename: str, revision: str = "main", endpoint: str = "https://huggingface.co"):
    endpoint = (endpoint or "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{repo_id}/resolve/{revision}/{filename}"


# ========== aria2 RPC（本工具自己管理的独立 aria2c 进程，不用 Trim 的） ==========

_aria2_daemon_process = None


def _shutdown_aria2_daemon():
    """
    主程序退出时（无论是 Ctrl+C、关闭窗口、还是被systemctl stop），
    主动结束我们自己拉起的 aria2c 子进程，避免留下孤儿进程占用带宽和端口。
    这个函数会被注册到 atexit 和信号处理器里，双重保险。
    """
    global _aria2_daemon_process
    if _aria2_daemon_process is not None and _aria2_daemon_process.poll() is None:
        try:
            _aria2_daemon_process.terminate()  # 先礼貌地发送终止信号，让 aria2c 有机会保存下载进度
            _aria2_daemon_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _aria2_daemon_process.kill()  # 5 秒还没退出就强制杀掉
        except Exception:
            pass
    try:
        if os.path.exists(ARIA2_PID_FILE):
            os.remove(ARIA2_PID_FILE)
    except Exception:
        pass


def _cleanup_stale_aria2_from_previous_run():
    """
    程序启动时调用一次：如果上一次运行时（比如被强制关闭窗口、没走到正常退出流程）
    残留了一个孤儿 aria2c 进程，这里把它杀掉，避免越攒越多、占用带宽和端口。
    """
    if not os.path.exists(ARIA2_PID_FILE):
        return
    try:
        with open(ARIA2_PID_FILE, "r") as f:
            old_pid = int(f.read().strip())
        os.kill(old_pid, signal.SIGTERM)
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        pass  # 进程本来就不存在了，或者已经是别的无关进程占了这个 PID，忽略即可
    finally:
        try:
            os.remove(ARIA2_PID_FILE)
        except Exception:
            pass


def _handle_termination_signal(signum, frame):
    """收到 Ctrl+C（SIGINT）或系统终止信号（SIGTERM）时，先清理子进程再退出"""
    _shutdown_aria2_daemon()
    sys.exit(0)


atexit.register(_shutdown_aria2_daemon)
signal.signal(signal.SIGINT, _handle_termination_signal)
signal.signal(signal.SIGTERM, _handle_termination_signal)


def _rpc_port_from_url(rpc_url: str) -> str:
    # 从 http://127.0.0.1:6801/jsonrpc 里提取端口号
    m = re.search(r':(\d+)/', rpc_url)
    return m.group(1) if m else "6801"


def is_aria2_daemon_alive(cfg) -> bool:
    try:
        aria2_call("aria2.getVersion", [], cfg=cfg)
        return True
    except Exception:
        return False


def _find_aria2_executable():
    """
    找 aria2c 可执行文件：
    1. 先看 PATH 里有没有（Linux 装了 apt install aria2 就在这）
    2. 再看程序自身所在目录（打包成 exe 时可以把 aria2c.exe 放在同一个文件夹）
    """
    import shutil as _shutil

    exe_name = "aria2c.exe" if platform.system() == "Windows" else "aria2c"

    found = _shutil.which(exe_name)
    if found:
        return found

    # PyInstaller 打包后，用 sys._MEIPASS 或 sys.executable 所在目录
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__)))
    local_path = os.path.join(base_dir, exe_name)
    if os.path.exists(local_path):
        return local_path

    return exe_name  # 找不到就原样返回，让 subprocess 报 FileNotFoundError


def ensure_aria2_daemon(cfg):
    """
    检查本工具专属的 aria2c RPC 守护进程是否在跑，
    如果没在跑就自己拉起一个（跟 Trim 的 aria2c 完全独立，端口/密钥都不同）。
    """
    global _aria2_daemon_process

    if is_aria2_daemon_alive(cfg):
        return True

    port = _rpc_port_from_url(cfg["rpc_url"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    cmd = [
        _find_aria2_executable(),
        "--enable-rpc",
        "--rpc-listen-all=false",  # 只监听本机，不对外网暴露
        f"--rpc-listen-port={port}",
        f"--rpc-secret={cfg['rpc_secret']}",
        f"--dir={cfg['save_dir']}",
        "--continue=true",
        "--max-concurrent-downloads=5",
        "--daemon=false",
    ]

    log_f = open(ARIA2_LOG_FILE, "a", encoding="utf-8")
    popen_kwargs = {"stdout": log_f, "stderr": subprocess.STDOUT}
    if platform.system() == "Windows":
        # Windows 没有 start_new_session，用 CREATE_NO_WINDOW 避免弹出黑窗口
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        popen_kwargs["start_new_session"] = True

    try:
        _aria2_daemon_process = subprocess.Popen(cmd, **popen_kwargs)
        try:
            with open(ARIA2_PID_FILE, "w") as f:
                f.write(str(_aria2_daemon_process.pid))
        except Exception:
            pass  # 写 PID 文件失败不影响主流程，只是下次启动少一层清理保险
    except FileNotFoundError:
        log_f.write(
            "\n[启动失败] 找不到 aria2c 可执行文件。\n"
            "Windows: 请下载 aria2 (https://github.com/aria2/aria2/releases)，\n"
            "解压后把 aria2c.exe 所在目录加入系统 PATH，或者放进本程序同一目录下。\n"
            "Linux: 请运行 apt install aria2 -y（或对应发行版的包管理器）。\n"
        )
        log_f.close()
        return False

    # 等待 RPC 端口起来，最多等 5 秒
    for _ in range(10):
        time.sleep(0.5)
        if is_aria2_daemon_alive(cfg):
            return True
    return False


def aria2_call(method: str, params=None, cfg=None):
    cfg = cfg or load_config()
    if params is None:
        params = []
    rpc_params = []
    if cfg.get("rpc_secret"):
        rpc_params.append(f"token:{cfg['rpc_secret']}")
    rpc_params.extend(params)
    payload = {
        "jsonrpc": "2.0",
        "id": "hf-downloader",
        "method": method,
        "params": rpc_params,
    }
    resp = requests.post(cfg["rpc_url"], json=payload, timeout=15)
    resp.raise_for_status()
    result = resp.json()
    if "error" in result:
        raise RuntimeError(result["error"])
    return result.get("result")


def aria2_add_download(url: str, save_dir: str, out_name: str, connections: int, cfg=None):
    options = {
        "dir": save_dir,
        "out": out_name,
        "split": str(connections),
        "max-connection-per-server": str(connections),
        "min-split-size": "1M",
        "continue": "true",
    }
    return aria2_call("aria2.addUri", [[url], options], cfg=cfg)


def aria2_status_summary(cfg=None):
    """汇总当前活跃/等待/最近完成的任务状态"""
    active = aria2_call("aria2.tellActive", [], cfg=cfg) or []
    waiting = aria2_call("aria2.tellWaiting", [0, 50], cfg=cfg) or []
    stopped = aria2_call("aria2.tellStopped", [0, 20], cfg=cfg) or []

    def simplify(task):
        total = int(task.get("totalLength", 0) or 0)
        done = int(task.get("completedLength", 0) or 0)
        speed = int(task.get("downloadSpeed", 0) or 0)
        name = ""
        files = task.get("files", [])
        if files:
            name = os.path.basename(files[0].get("path", "") or "")
        pct = round(done / total * 100, 1) if total > 0 else 0
        return {
            "gid": task.get("gid"),
            "name": name,
            "status": task.get("status"),
            "totalLength": total,
            "completedLength": done,
            "downloadSpeed": speed,
            "percent": pct,
        }

    return {
        "active": [simplify(t) for t in active],
        "waiting": [simplify(t) for t in waiting],
        "stopped": [simplify(t) for t in stopped],
    }


# ========== 路由 ==========

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>HuggingFace 模型批量下载工具</title>
<style>
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; max-width: 900px; margin: 30px auto; padding: 0 20px; color: #222; }
  h1 { font-size: 22px; }
  .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px 20px; margin-bottom: 18px; }
  label { display:block; font-size: 13px; color:#555; margin-bottom:4px; }
  input[type=text] { width: 100%; padding: 8px; box-sizing: border-box; border:1px solid #ccc; border-radius:4px; font-size:14px; }
  button { background:#2b7de9; color:#fff; border:none; padding:9px 18px; border-radius:5px; cursor:pointer; font-size:14px; }
  button:hover { background:#1f66c2; }
  button.secondary { background:#888; }
  button.secondary:hover { background:#666; }
  .file-list { max-height: 360px; overflow-y:auto; border:1px solid #eee; border-radius:4px; padding:8px 12px; margin-top:10px;}
  .file-row { display:flex; align-items:center; padding:4px 0; border-bottom:1px solid #f4f4f4; font-size:13px;}
  .file-row:last-child{border-bottom:none;}
  .file-row input { margin-right:8px; }
  .file-size { margin-left:auto; color:#999; font-size:12px; }
  .toolbar { display:flex; gap:8px; margin: 10px 0; align-items:center;}
  .status-row { font-size:13px; padding:6px 0; border-bottom:1px solid #f0f0f0; }
  .bar-bg { background:#eee; border-radius:4px; height:8px; overflow:hidden; margin-top:4px;}
  .bar-fg { background:#2b7de9; height:100%; }
  .muted { color:#999; font-size:12px; }
  .row2 { display:flex; gap:12px; }
  .row2 > div { flex:1; }
  #log { font-family: monospace; font-size:12px; color:#555; margin-top:8px; white-space:pre-wrap; }
</style>
</head>
<body>

<h1>🤗 HuggingFace 模型批量下载工具</h1>
<p class="muted">运行在飞牛NAS本地，通过 Trim (aria2 RPC) 后端多线程下载</p>

<div class="card">
  <label>模型仓库 ID（如 JonathanColetti/Qwen3.8-27B-Uncensored-GGUF）</label>
  <input type="text" id="repoId" placeholder="用户名/仓库名">

  <label style="margin-top:12px;">下载源地址（可选，留空默认用官网 huggingface.co；国外访问慢可以填国内镜像站，如 https://hf-mirror.com）</label>
  <input type="text" id="hfEndpoint" value="{{ cfg.hf_endpoint }}" placeholder="https://huggingface.co">
  <div class="toolbar">
    <button onclick="loadFiles()">加载文件列表</button>
    <span id="loadStatus" class="muted"></span>
  </div>
  <div id="fileListWrap" style="display:none;">
    <div class="toolbar">
      <button class="secondary" onclick="toggleAll(true)">全选</button>
      <button class="secondary" onclick="toggleAll(false)">全不选</button>
      <span id="fileCount" class="muted"></span>
    </div>
    <div class="file-list" id="fileList"></div>
  </div>
</div>

<div class="card">
  <div class="row2">
    <div>
      <label>保存目录</label>
      <div style="display:flex; gap:8px;">
        <input type="text" id="saveDir" value="{{ cfg.save_dir }}" style="flex:1;">
        <button class="secondary" onclick="openBrowseModal()" style="white-space:nowrap;">浏览...</button>
      </div>
    </div>
    <div>
      <label>单文件并发连接数（1-16）</label>
      <input type="text" id="connections" value="{{ cfg.connections_per_file }}">
    </div>
  </div>
  <div class="toolbar">
    <button onclick="submitDownload()">提交下载</button>
    <span id="submitStatus" class="muted"></span>
  </div>
</div>

<!-- 目录浏览弹窗 -->
<div id="browseModal" style="display:none; position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.4); z-index:100;">
  <div style="background:#fff; max-width:600px; margin:60px auto; border-radius:8px; padding:20px; max-height:70vh; display:flex; flex-direction:column;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
      <strong id="browseCurrentPath">/vol1</strong>
      <span style="cursor:pointer; font-size:18px;" onclick="closeBrowseModal()">&times;</span>
    </div>
    <div id="browseList" style="overflow-y:auto; flex:1; border:1px solid #eee; border-radius:4px; padding:8px;"></div>
    <div class="toolbar" style="margin-top:12px;">
      <button onclick="selectCurrentBrowseDir()">选择当前目录</button>
      <button class="secondary" onclick="closeBrowseModal()">取消</button>
    </div>
  </div>
</div>

<div class="card">
  <label>RPC 设置（本工具自带独立 aria2c 进程，密钥自动生成，一般不需要改动）</label>
  <div class="row2">
    <div>
      <input type="text" id="rpcUrl" value="{{ cfg.rpc_url }}" placeholder="RPC 地址">
    </div>
    <div>
      <input type="text" id="rpcSecret" value="{{ cfg.rpc_secret }}" placeholder="RPC 密钥">
    </div>
  </div>
  <div class="toolbar">
    <button class="secondary" onclick="saveSettings()">保存设置</button>
    <span id="settingsStatus" class="muted"></span>
  </div>
</div>

<div class="card">
  <label>当前下载任务</label>
  <div id="statusList"></div>
  <div class="toolbar">
    <button class="secondary" onclick="refreshStatus()">刷新状态</button>
    <span class="muted">每 5 秒自动刷新一次</span>
  </div>
</div>

<script>
let currentFiles = [];

async function syncEndpoint() {
  const endpoint = document.getElementById('hfEndpoint').value.trim();
  try {
    await fetch('/api/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({hf_endpoint: endpoint})
    });
  } catch (e) { /* 静默失败，不影响主流程 */ }
}

async function loadFiles() {
  const repoId = document.getElementById('repoId').value.trim();
  const statusEl = document.getElementById('loadStatus');
  if (!repoId) { statusEl.textContent = '请先输入仓库ID'; return; }
  statusEl.textContent = '正在加载...';
  await syncEndpoint();
  try {
    const res = await fetch('/api/files?repo_id=' + encodeURIComponent(repoId));
    const data = await res.json();
    if (data.error) { statusEl.textContent = '错误: ' + data.error; return; }
    currentFiles = data.files;
    renderFileList();
    document.getElementById('fileListWrap').style.display = 'block';
    document.getElementById('fileCount').textContent = '共 ' + currentFiles.length + ' 个文件';
    statusEl.textContent = '加载成功';
  } catch (e) {
    statusEl.textContent = '请求失败: ' + e;
  }
}

function humanSize(bytes) {
  if (!bytes && bytes !== 0) return '未知大小';
  const units = ['B','KB','MB','GB','TB'];
  let i = 0; let n = bytes;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return n.toFixed(1) + ' ' + units[i];
}

function renderFileList() {
  const wrap = document.getElementById('fileList');
  wrap.innerHTML = '';
  currentFiles.forEach((f, idx) => {
    const row = document.createElement('div');
    row.className = 'file-row';
    row.innerHTML = `
      <input type="checkbox" checked data-idx="${idx}">
      <span>${f.name}</span>
      <span class="file-size">${humanSize(f.size)}</span>
    `;
    wrap.appendChild(row);
  });
}

function toggleAll(checked) {
  document.querySelectorAll('#fileList input[type=checkbox]').forEach(cb => cb.checked = checked);
}

async function submitDownload() {
  const repoId = document.getElementById('repoId').value.trim();
  const saveDir = document.getElementById('saveDir').value.trim();
  const connections = parseInt(document.getElementById('connections').value.trim() || '16', 10);
  const statusEl = document.getElementById('submitStatus');

  const selected = [];
  document.querySelectorAll('#fileList input[type=checkbox]').forEach(cb => {
    if (cb.checked) selected.push(currentFiles[parseInt(cb.dataset.idx, 10)].name);
  });

  if (!repoId || selected.length === 0) {
    statusEl.textContent = '请先加载文件列表并勾选文件';
    return;
  }

  await syncEndpoint();

  statusEl.textContent = '提交中...';
  try {
    const res = await fetch('/api/download', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({repo_id: repoId, files: selected, save_dir: saveDir, connections: connections})
    });
    const data = await res.json();
    if (data.error) { statusEl.textContent = '错误: ' + data.error; return; }
    statusEl.textContent = `已提交 ${data.submitted} 个任务`;
    refreshStatus();
  } catch (e) {
    statusEl.textContent = '提交失败: ' + e;
  }
}

async function saveSettings() {
  const rpcUrl = document.getElementById('rpcUrl').value.trim();
  const rpcSecret = document.getElementById('rpcSecret').value.trim();
  const statusEl = document.getElementById('settingsStatus');
  statusEl.textContent = '保存中...';
  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({rpc_url: rpcUrl, rpc_secret: rpcSecret})
    });
    const data = await res.json();
    statusEl.textContent = data.ok ? '已保存' : ('失败: ' + data.error);
  } catch (e) {
    statusEl.textContent = '保存失败: ' + e;
  }
}

function renderStatusGroup(title, tasks) {
  if (!tasks.length) return '';
  let html = `<div class="muted" style="margin-top:8px;">${title}</div>`;
  tasks.forEach(t => {
    let buttons = '';
    if (t.status === 'active') {
      buttons = `
        <button class="secondary" style="padding:3px 10px; font-size:12px;" onclick="pauseTask('${t.gid}')">暂停</button>
        <button class="secondary" style="padding:3px 10px; font-size:12px;" onclick="cancelTask('${t.gid}')">取消</button>
      `;
    } else if (t.status === 'paused') {
      buttons = `
        <button class="secondary" style="padding:3px 10px; font-size:12px;" onclick="resumeTask('${t.gid}')">继续</button>
        <button class="secondary" style="padding:3px 10px; font-size:12px;" onclick="cancelTask('${t.gid}')">取消</button>
      `;
    } else if (t.status === 'waiting') {
      buttons = `<button class="secondary" style="padding:3px 10px; font-size:12px;" onclick="cancelTask('${t.gid}')">取消</button>`;
    }
    html += `
      <div class="status-row">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span>${t.name || '(未知文件名)'} — ${t.status} — ${(t.downloadSpeed/1024).toFixed(0)} KB/s — ${t.percent}%</span>
          <span style="display:flex; gap:6px;">${buttons}</span>
        </div>
        <div class="bar-bg"><div class="bar-fg" style="width:${t.percent}%;"></div></div>
      </div>
    `;
  });
  return html;
}

async function pauseTask(gid) {
  await fetch('/api/task/pause', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({gid})
  });
  refreshStatus();
}

async function resumeTask(gid) {
  await fetch('/api/task/resume', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({gid})
  });
  refreshStatus();
}

async function cancelTask(gid) {
  if (!confirm('确定要取消这个下载任务吗？已下载的部分不会被删除，下次可以重新提交继续下载。')) return;
  await fetch('/api/task/cancel', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({gid})
  });
  refreshStatus();
}

async function refreshStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.error) {
      document.getElementById('statusList').innerHTML = '<span class="muted">状态获取失败: ' + data.error + '</span>';
      return;
    }
    let html = '';
    html += renderStatusGroup('下载中', data.active);
    html += renderStatusGroup('等待中', data.waiting);
    html += renderStatusGroup('最近完成/停止', data.stopped);
    document.getElementById('statusList').innerHTML = html || '<span class="muted">暂无任务</span>';
  } catch (e) {
    document.getElementById('statusList').innerHTML = '<span class="muted">状态获取失败</span>';
  }
}

let browseCurrentPath = '';

async function openBrowseModal() {
  document.getElementById('browseModal').style.display = 'block';
  const startPath = document.getElementById('saveDir').value.trim();
  await loadBrowseDir(startPath);
}

function closeBrowseModal() {
  document.getElementById('browseModal').style.display = 'none';
}

async function loadBrowseDir(path) {
  try {
    const res = await fetch('/api/browse?path=' + encodeURIComponent(path));
    const data = await res.json();
    if (data.error) {
      // 路径无效时退回到根目录（空字符串让后端决定：Linux 是 /，Windows 是盘符列表）
      if (path !== '') { await loadBrowseDir(''); return; }
      document.getElementById('browseList').innerHTML = '<div class="muted">' + data.error + '</div>';
      return;
    }
    browseCurrentPath = data.is_drive_list ? '' : data.path;
    document.getElementById('browseCurrentPath').textContent = data.is_drive_list ? '选择磁盘' : data.path;
    let html = '';
    if (data.parent) {
      html += `<div style="padding:6px; cursor:pointer; color:#2b7de9;" onclick="loadBrowseDir('${data.parent.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}')">.. (上一级)</div>`;
    }
    data.dirs.forEach(d => {
      let full, label;
      if (data.is_drive_list) {
        full = d;       // Windows 盘符本身就是完整路径，如 "C:\"
        label = d;
      } else {
        full = (data.path.endsWith('/') || data.path.endsWith('\\')) ? data.path + d : data.path + '/' + d;
        label = d;
      }
      html += `<div style="padding:6px; cursor:pointer;" onclick="loadBrowseDir('${full.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}')">📁 ${label}</div>`;
    });
    if (!data.dirs.length && !data.parent) {
      html += '<div class="muted" style="padding:6px;">（没有子目录）</div>';
    }
    document.getElementById('browseList').innerHTML = html;
  } catch (e) {
    document.getElementById('browseList').innerHTML = '<div class="muted">加载失败: ' + e + '</div>';
  }
}

function selectCurrentBrowseDir() {
  if (!browseCurrentPath) {
    alert('请先点击进入一个具体的文件夹，再选择');
    return;
  }
  document.getElementById('saveDir').value = browseCurrentPath;
  closeBrowseModal();
}

refreshStatus();
setInterval(refreshStatus, 5000);
</script>

</body>
</html>
"""


@app.route("/")
def index():
    cfg = load_config()
    return render_template_string(INDEX_HTML, cfg=cfg)


@app.route("/api/files")
def api_files():
    repo_id = request.args.get("repo_id", "").strip()
    repo_id = normalize_repo_id(repo_id)
    if not repo_id:
        return jsonify({"error": "缺少 repo_id"}), 400
    cfg = load_config()
    try:
        files = hf_list_files(repo_id, endpoint=cfg.get("hf_endpoint"))
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download", methods=["POST"])
def api_download():
    data = request.get_json(force=True)
    repo_id = data.get("repo_id", "").strip()
    repo_id = normalize_repo_id(repo_id)
    files = data.get("files", [])
    save_dir = data.get("save_dir", "").strip()
    connections = int(data.get("connections", 16) or 16)
    connections = max(1, min(16, connections))

    if not repo_id or not files or not save_dir:
        return jsonify({"error": "参数不完整"}), 400

    cfg = load_config()
    cfg = ensure_secret(cfg)
    if not ensure_aria2_daemon(cfg):
        return jsonify({"error": "aria2 后台进程启动失败，请查看 aria2-daemon.log"}), 500

    submitted = 0
    errors = []
    for fname in files:
        try:
            url = hf_resolve_url(repo_id, fname, endpoint=cfg.get("hf_endpoint"))
            # 文件名可能带子目录，out 参数保留原始相对路径，aria2 会自动建子目录
            aria2_add_download(url, save_dir, fname, connections, cfg=cfg)
            submitted += 1
        except Exception as e:
            errors.append(f"{fname}: {e}")

    result = {"submitted": submitted}
    if errors:
        result["errors"] = errors
    return jsonify(result)


@app.route("/api/status")
def api_status():
    cfg = load_config()
    cfg = ensure_secret(cfg)
    if not is_aria2_daemon_alive(cfg):
        ensure_aria2_daemon(cfg)
    try:
        return jsonify(aria2_status_summary(cfg=cfg))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/task/pause", methods=["POST"])
def api_task_pause():
    data = request.get_json(force=True)
    gid = data.get("gid", "").strip()
    if not gid:
        return jsonify({"ok": False, "error": "缺少 gid"}), 400
    cfg = load_config()
    try:
        aria2_call("aria2.pause", [gid], cfg=cfg)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/task/resume", methods=["POST"])
def api_task_resume():
    data = request.get_json(force=True)
    gid = data.get("gid", "").strip()
    if not gid:
        return jsonify({"ok": False, "error": "缺少 gid"}), 400
    cfg = load_config()
    try:
        aria2_call("aria2.unpause", [gid], cfg=cfg)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/task/cancel", methods=["POST"])
def api_task_cancel():
    """
    取消一个任务。注意：这里只是让 aria2 停止这个下载任务，
    磁盘上已下载的部分和 .aria2 断点续传文件不会被删除——
    如果之后想接着下同一个文件，重新提交同样的下载任务即可断点续传。
    """
    data = request.get_json(force=True)
    gid = data.get("gid", "").strip()
    if not gid:
        return jsonify({"ok": False, "error": "缺少 gid"}), 400
    cfg = load_config()
    try:
        aria2_call("aria2.forceRemove", [gid], cfg=cfg)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True)
    cfg = load_config()
    if "rpc_url" in data and data["rpc_url"]:
        cfg["rpc_url"] = data["rpc_url"]
    if "rpc_secret" in data:
        cfg["rpc_secret"] = data["rpc_secret"]
    if "hf_endpoint" in data:
        endpoint = (data["hf_endpoint"] or "").strip()
        cfg["hf_endpoint"] = endpoint if endpoint else "https://huggingface.co"
    try:
        save_config(cfg)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/browse")
def api_browse():
    """
    列出指定路径下的子目录，供前端"浏览目录"弹窗使用。
    Linux/NAS：从 / 根目录开始，屏蔽几个系统虚拟目录。
    Windows：path 为空或 "ROOT" 时返回所有盘符（C:\\、D:\\ 等）供选择。
    """
    is_windows = platform.system() == "Windows"
    BLOCKED_PREFIXES = ("/proc", "/sys", "/dev", "/run")

    raw_path = request.args.get("path", "").strip()

    # Windows 下的"根目录"是盘符列表，不是单个路径
    if is_windows and (not raw_path or raw_path == "ROOT"):
        import string
        drives = []
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                drives.append(drive)
        return jsonify({"path": "ROOT", "parent": None, "dirs": drives, "is_drive_list": True})

    path = raw_path or ("/" if not is_windows else "ROOT")
    if not is_windows:
        path = os.path.normpath(path)
        if any(path == p or path.startswith(p + "/") for p in BLOCKED_PREFIXES):
            path = "/"

    if not os.path.isdir(path):
        return jsonify({"error": f"目录不存在: {path}"}), 400

    try:
        entries = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isdir(full) and not name.startswith("."):
                entries.append(name)

        if is_windows:
            # 盘符根目录（如 C:\）的上一级是"返回盘符列表"
            is_drive_root = len(path) <= 3 and path[1:3] in (":\\", ":/")
            parent = "ROOT" if is_drive_root else os.path.dirname(path)
        else:
            parent = os.path.dirname(path) if path != "/" else None

        return jsonify({"path": path, "parent": parent, "dirs": entries})
    except PermissionError:
        return jsonify({"error": "没有权限访问该目录"}), 403
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    _cleanup_stale_aria2_from_previous_run()
    _startup_cfg = ensure_secret(load_config())
    ensure_aria2_daemon(_startup_cfg)

    # 打包成 exe 双击运行时，自动打开浏览器，不用用户自己去输网址
    if getattr(sys, "frozen", False):
        import threading
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5678")).start()

    app.run(host="0.0.0.0", port=5678, debug=False)
