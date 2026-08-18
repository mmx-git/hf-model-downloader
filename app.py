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

import json
import os
import re
import secrets
import subprocess
import time

import requests
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# ========== 默认配置（也可以在网页"设置"里覆盖，会写入 config.json） ==========
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
ARIA2_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aria2-daemon.log")

DEFAULT_CONFIG = {
    "rpc_url": "http://127.0.0.1:6801/jsonrpc",  # 本工具自建的独立 aria2 实例，跟 Trim 的 6800 分开
    "rpc_secret": "",  # 首次启动自动生成，不用手动填
    "save_dir": "/vol1/1000/download",
    "connections_per_file": 16,  # aria2 单文件最大连接数，1-16
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
    - 完整链接: "https://huggingface.co/user/model/tree/main"
    - markdown 链接: "[user/model](https://huggingface.co/user/model/tree/main)"
    """
    raw = raw.strip()
    # markdown 链接格式，取括号里的 URL
    md_match = re.search(r'\]\((https?://[^\)]+)\)', raw)
    if md_match:
        raw = md_match.group(1)
    # 完整 URL，提取 huggingface.co 后面的 user/model 部分
    url_match = re.search(r'huggingface\.co/([^/\s]+/[^/\s\)\]]+)', raw)
    if url_match:
        return url_match.group(1)
    # 已经是纯 repo_id 格式
    return raw


def hf_list_files(repo_id: str, revision: str = "main"):
    """调用 HF 的 tree API 获取仓库文件列表（这个接口会带上文件大小，
    普通的 /api/models/{repo_id} 接口不带 size 字段）"""
    url = f"https://huggingface.co/api/models/{repo_id}/tree/{revision}"
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


def hf_resolve_url(repo_id: str, filename: str, revision: str = "main"):
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


# ========== aria2 RPC（本工具自己管理的独立 aria2c 进程，不用 Trim 的） ==========

_aria2_daemon_process = None


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
        "aria2c",
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
    _aria2_daemon_process = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

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

INDEX_HTML = """
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

async function loadFiles() {
  const repoId = document.getElementById('repoId').value.trim();
  const statusEl = document.getElementById('loadStatus');
  if (!repoId) { statusEl.textContent = '请先输入仓库ID'; return; }
  statusEl.textContent = '正在加载...';
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
    html += `
      <div class="status-row">
        <div>${t.name || '(未知文件名)'} — ${t.status} — ${(t.downloadSpeed/1024).toFixed(0)} KB/s — ${t.percent}%</div>
        <div class="bar-bg"><div class="bar-fg" style="width:${t.percent}%;"></div></div>
      </div>
    `;
  });
  return html;
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

let browseCurrentPath = '/vol1';

async function openBrowseModal() {
  document.getElementById('browseModal').style.display = 'block';
  const startPath = document.getElementById('saveDir').value.trim() || '/vol1';
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
      // 路径无效时退回到根目录
      if (path !== '/') { await loadBrowseDir('/'); return; }
      document.getElementById('browseList').innerHTML = '<div class="muted">' + data.error + '</div>';
      return;
    }
    browseCurrentPath = data.path;
    document.getElementById('browseCurrentPath').textContent = data.path;
    let html = '';
    if (data.parent) {
      html += `<div style="padding:6px; cursor:pointer; color:#2b7de9;" onclick="loadBrowseDir('${data.parent.replace(/'/g, "\\'")}')">.. (上一级)</div>`;
    }
    data.dirs.forEach(d => {
      const full = (data.path.endsWith('/') ? data.path : data.path + '/') + d;
      html += `<div style="padding:6px; cursor:pointer;" onclick="loadBrowseDir('${full.replace(/'/g, "\\'")}')">📁 ${d}</div>`;
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
    try:
        files = hf_list_files(repo_id)
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
            url = hf_resolve_url(repo_id, fname)
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


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True)
    cfg = load_config()
    if "rpc_url" in data and data["rpc_url"]:
        cfg["rpc_url"] = data["rpc_url"]
    if "rpc_secret" in data:
        cfg["rpc_secret"] = data["rpc_secret"]
    try:
        save_config(cfg)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/browse")
def api_browse():
    """
    列出指定路径下的子目录，供前端"浏览目录"弹窗使用。
    允许浏览整个文件系统（方便跳转到 /vol1、/vol2 等不同存储卷），
    只屏蔽几个 Linux 系统虚拟目录，避免误选到无意义的路径。
    """
    BLOCKED_PREFIXES = ("/proc", "/sys", "/dev", "/run")

    path = request.args.get("path", "/").strip() or "/"
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
        parent = os.path.dirname(path) if path != "/" else None
        return jsonify({"path": path, "parent": parent, "dirs": entries})
    except PermissionError:
        return jsonify({"error": "没有权限访问该目录"}), 403
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    _startup_cfg = ensure_secret(load_config())
    ensure_aria2_daemon(_startup_cfg)
    app.run(host="0.0.0.0", port=5678, debug=False)
