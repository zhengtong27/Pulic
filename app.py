# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template_string
import re
import os
from openai import OpenAI
from dashscope import Application

app = Flask(__name__)

# ============================================================
# 环境变量读取（在 Render 的 Environment 中设置）
# ============================================================
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
DASHSCOPE_WORKSPACE_ID = os.environ.get("DASHSCOPE_WORKSPACE_ID")  # 可选，如果知识库在默认空间则不需要
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if not DASHSCOPE_API_KEY:
    print("警告：未设置环境变量 DASHSCOPE_API_KEY，RAG 检索将不可用，大模型调用也可能失败")

# 初始化 OpenAI 客户端（用于大模型生成）
if DASHSCOPE_API_KEY:
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
else:
    client = None
    print("错误：无法创建 OpenAI 客户端，请设置环境变量 DASHSCOPE_API_KEY")

# 模型名称（建议使用稳定的公开模型）
MODEL_NAME = "qwen3-32b_eb56be00"  
# ============================================================
# 知识库检索函数
# ============================================================
def retrieve_from_knowledge_base(question, top_k=3):
    """
    从阿里云百炼知识库中检索与问题最相关的文档片段
    参数:
        question: 用户问题
        top_k: 返回的最相关片段数量（默认3）
    返回:
        list of str: 文档片段列表
    """
    if not DASHSCOPE_API_KEY:
        return []
    try:
        INDEX_NAME = "ssy8053dlh"   
        
        response = Retrieval.search(
            workspace_id=DASHSCOPE_WORKSPACE_ID,  
            index_name=INDEX_NAME,
            query=question,
            dense_similarity_top_k=top_k,
            sparse_similarity_top_k=top_k,
            enable_reranking=True,
            rerank_top_n=top_k
        )
        docs = []
        if response and hasattr(response, 'output') and response.output:
            for doc in response.output.documents:
                docs.append(doc.text)
        return docs
    except Exception as e:
        print(f"知识库检索失败: {e}")
        return []

# ============================================================
# 大模型调用函数（集成 RAG 检索）
# ============================================================
def call_llm(question):
    # 1. 非紧急症状过滤（保持原有安全逻辑）
    mild_pattern = re.compile(
        r'(头(?:有?点)?痛|头(?:有?点)?晕|眼花|疲劳|乏力|失眠|焦虑|消化不良|颈部不适|有点不舒服)',
        re.IGNORECASE
    )
    if mild_pattern.search(question):
        return ("头痛的原因很多，比如疲劳、紧张或血压波动。请先坐下休息，喝点温水，观察一下。"
                "如果疼痛持续不缓解或加重，再咨询医生。注意：本内容仅供参考，如有需要请及时就医。")

    # 2. 调用百炼 RAG 应用（自动检索知识库并生成回答）
    try:
        response = Application.call(
            app_id=os.environ.get("DASHSCOPE_APP_ID"),      # 在 Render 环境变量中设置
            prompt=question,
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            workspace_id=os.environ.get("DASHSCOPE_WORKSPACE_ID")  # 可选，如果知识库在非默认业务空间则需要
        )
        # 提取回答内容
        full_answer = response.output.text if response and response.output else "抱歉，未能获取到有效回答。"

    except Exception as e:
        print(f"百炼应用调用失败: {e}")
        return "抱歉，系统繁忙，请稍后再试。"

    # 3. 后处理：去除常见的肯定性开头短语
    prefix_pattern = re.compile(r'^(您说得对|好的|是的|没错|嗯|对，|对的，|好的，)\s*', re.IGNORECASE)
    full_answer = prefix_pattern.sub('', full_answer).strip()

    # 4. 二次安全过滤：如果模型错误地输出了紧急关键词，且问题属于非紧急，则覆盖回答
    emergency_keywords = ["脑卒中", "中风", "拨打120", "紧急就医", "立即前往医院", "专业医生进行评估"]
    if full_answer and any(kw in full_answer for kw in emergency_keywords):
        if mild_pattern.search(question):
            return ("头痛的原因很多，比如疲劳、紧张或血压波动。请先坐下休息，喝点温水，观察一下。"
                    "如果疼痛持续不缓解或加重，再咨询医生。注意：本内容仅供参考，如有需要请及时就医。")
    return full_answer if full_answer else "抱歉，模型未返回有效回答。"

# ============================================================
# Flask 路由
# ============================================================
@app.after_request
def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/api/switch_lang', methods=['POST', 'OPTIONS'])
def switch_lang():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({"status": "success"})

@app.route('/api/stroke_qa', methods=['POST', 'OPTIONS'])
def stroke_qa():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    if not question:
        return jsonify({"status": "error", "message": "问题不能为空"})
    answer = call_llm(question)
    return jsonify({"status": "success", "data": {"answer": answer}})

@app.route('/')
def index():
    return render_template_string('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes, viewport-fit=cover">
    <title>福医卒中通 · 脑卒中智能诊疗助手</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        :root {
            --font-scale: 1;
        }
        body {
            background: linear-gradient(135deg, #f0f7ff 0%, #e6f4fd 100%);
            padding: 20px;
            min-height: 100vh;
            font-size: calc(16px * var(--font-scale));
            padding-bottom: 90px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            margin-bottom: 20px;
        }
        header h1 {
            font-size: calc(56px * var(--font-scale));
            color: #0077cc;
            margin-bottom: 6px;
        }
        header p {
            color: #666;
            font-size: calc(28px * var(--font-scale));
        }
        .main-card {
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0,100,200,0.08);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: calc(100vh - 40px - 90px);
        }
        .chat-header-bar {
            background: linear-gradient(90deg, #0077cc, #0099ee);
            color: #fff;
            padding: 14px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: relative;
        }
        .chat-header-bar h2 {
            font-size: calc(32px * var(--font-scale));
            font-weight: 500;
        }
        .header-btns {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .header-btn {
            padding: 6px 12px;
            border-radius: 20px;
            background: rgba(255,255,255,0.2);
            color: #fff;
            border: none;
            font-size: calc(26px * var(--font-scale));
            cursor: pointer;
            white-space: nowrap;
            -webkit-tap-highlight-color: transparent;
            touch-action: manipulation;
        }
        .font-modal {
            position: absolute;
            top: 60px;
            right: 0;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            padding: 15px;
            z-index: 1000;
            display: none;
            min-width: 280px;
        }
        .font-modal.show {
            display: block;
        }
        .font-modal .modal-title {
            color: #0077cc;
            font-size: calc(24px * var(--font-scale));
            margin-bottom: 10px;
            font-weight: 500;
        }
        .font-modal .opt-group {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            align-items: center;
        }
        .font-modal .opt-btn {
            flex: 1;
            padding: 8px 0;
            border: 1px solid #eee;
            border-radius: 6px;
            background: #f8fcff;
            color: #0077cc;
            font-size: calc(22px * var(--font-scale));
            cursor: pointer;
        }
        .font-modal .opt-btn.active {
            background: #0077cc;
            color: #fff;
            border-color: #0077cc;
        }
        .font-modal .input-group {
            display: flex;
            gap: 8px;
            align-items: center;
            margin-bottom: 10px;
        }
        .font-modal input {
            flex: 1;
            padding: 8px 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            outline: none;
            font-size: calc(22px * var(--font-scale));
        }
        .font-modal .confirm-btn {
            width: 100%;
            padding: 8px 0;
            border: none;
            border-radius: 6px;
            background: #0077cc;
            color: #fff;
            font-size: calc(22px * var(--font-scale));
            cursor: pointer;
        }
        .font-modal .tip-text {
            font-size: calc(20px * var(--font-scale));
            color: #666;
            margin-top: 8px;
            text-align: center;
        }
        .chat-content {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        .sidebar {
            width: 200px;
            background: #f8fcff;
            border-right: 1px solid #e6f7ff;
            padding: 20px;
            overflow-y: auto;
        }
        .avatar-box {
            width: 100px;
            height: 100px;
            margin: 0 auto 10px;
            border-radius: 50%;
            background: #fff;
            border: 3px solid #0077cc;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: breathing 4s infinite ease-in-out;
        }
        .avatar-box.patient {
            border-color: #5499c7;
        }
        .avatar-card {
            text-align: center;
            margin-bottom: 24px;
        }
        .avatar-card h3 {
            font-size: calc(28px * var(--font-scale));
            color: #0077cc;
        }
        .patient-text {
            color: #5499c7 !important;
        }
        @keyframes breathing {
            0%,100%{transform: scale(1);}
            50%{transform: scale(1.04);}
        }
        .chat-main {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .quick-questions {
            padding: 12px 16px;
            background: #f5f9fe;
            border-bottom: 1px solid #e6f0fa;
            overflow-x: auto;
            white-space: nowrap;
            display: flex;
            gap: 10px;
            scrollbar-width: thin;
            -webkit-overflow-scrolling: touch;
        }
        .quick-questions::-webkit-scrollbar {
            height: 4px;
        }
        .quick-questions button {
            background: #eef3fc;
            border: 1px solid #cce4f5;
            border-radius: 24px;
            padding: 8px 16px;
            font-size: calc(24px * var(--font-scale));
            color: #0077cc;
            cursor: pointer;
            white-space: nowrap;
            flex-shrink: 0;
            transition: all 0.2s;
            -webkit-tap-highlight-color: transparent;
            touch-action: manipulation;
        }
        .quick-questions button:hover {
            background: #0077cc;
            color: #fff;
            border-color: #0077cc;
        }
        .chat-body {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #fafbfc;
        }
        .message {
            display: flex;
            gap: 10px;
            margin-bottom: 14px;
            max-width: 75%;
        }
        .message.user {
            margin-left: auto;
            flex-direction: row-reverse;
        }
        .msg-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #fff;
            border: 1px solid #eee;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .msg-bubble {
            padding: 10px 14px;
            border-radius: 14px;
            background: #fff;
            border: 1px solid #eee;
            line-height: 1.5;
            font-size: calc(28px * var(--font-scale));
            word-break: break-word;
        }
        .message.user .msg-bubble {
            background: #0077cc;
            color: #fff;
            border: none;
        }
        .chat-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 14px 20px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
            align-items: center;
            background: #fff;
            z-index: 99;
            box-shadow: 0 -2px 10px rgba(0,100,200,0.05);
            padding-bottom: calc(14px + env(safe-area-inset-bottom));
        }
        .chat-input {
            flex: 1;
            padding: 10px 16px;
            border: 1px solid #ddd;
            border-radius: 24px;
            outline: none;
            font-size: calc(28px * var(--font-scale));
            min-width: 0;
        }
        .send-btn, .clear-btn {
            padding: 10px 18px;
            border-radius: 24px;
            background: #0077cc;
            color: #fff;
            border: none;
            cursor: pointer;
            font-size: calc(28px * var(--font-scale));
            white-space: nowrap;
            -webkit-tap-highlight-color: transparent;
            touch-action: manipulation;
        }
        .clear-btn {
            background: #777;
        }
        .mic-btn {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: #0077cc;
            color: #fff;
            font-size: calc(32px * var(--font-scale));
            border: none;
            cursor: pointer;
            flex-shrink: 0;
            -webkit-tap-highlight-color: transparent;
            touch-action: manipulation;
        }
        .mic-btn.recording {
            background: #e53935;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%{transform: scale(1);}
            50%{transform: scale(1.1);}
            100%{transform: scale(1);}
        }
        .modal-mask {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 999;
            background: rgba(0,0,0,0.5);
            display: none;
        }
        .modal-mask.show {
            display: block;
        }

        /* 移动端适配 */
        @media (max-width: 768px) {
            .sidebar { display: none !important; }
            .chat-main { width: 100% !important; }
            body { padding: 10px 0 90px 0 !important; }
            .message { max-width: 90% !important; }
            .msg-bubble { font-size: calc(32px * var(--font-scale)) !important; padding: 8px 12px !important; }
            .quick-questions button { font-size: calc(26px * var(--font-scale)) !important; padding: 10px 16px !important; }
            .header-btn { font-size: calc(24px * var(--font-scale)) !important; padding: 6px 10px !important; }
            header h1 { font-size: calc(44px * var(--font-scale)) !important; }
            .chat-header-bar h2 { font-size: calc(28px * var(--font-scale)) !important; }
            .chat-footer { padding: 8px 12px !important; }
            .send-btn, .clear-btn { font-size: calc(28px * var(--font-scale)) !important; padding: 8px 14px !important; }
            .mic-btn { width: 44px !important; height: 44px !important; font-size: 24px !important; }
        }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>福医卒中通 · 脑卒中智能诊疗助手</h1>
        <p>专业脑卒中科普 | 智能问答咨询</p>
    </header>
    <div class="main-card">
        <div class="chat-header-bar">
            <h2>脑卒中智能咨询</h2>
            <div class="header-btns">
                <button class="header-btn" id="langBtn">切换英文</button>
                <button class="header-btn" id="fontBtn">字体调节</button>
                <button class="header-btn" id="voiceBtn">语音播报：开</button>
                <div class="font-modal" id="fontModal">
                    <div class="modal-title">字体大小调节</div>
                    <div class="opt-group">
                        <button class="opt-btn active" id="enlargeBtn">放大</button>
                        <button class="opt-btn" id="narrowBtn">缩小</button>
                    </div>
                    <div class="input-group">
                        <input type="number" id="scaleInput" placeholder="请输入倍数" step="0.5" value="">
                    </div>
                    <button class="confirm-btn" id="confirmFontBtn">确认调节</button>
                    <div class="tip-text">放大：1-4（步长0.5）| 缩小：0.3-1（步长0.1）</div>
                </div>
            </div>
        </div>
        <div class="chat-content">
            <div class="sidebar">
                <div class="avatar-card">
                    <div class="avatar-box">
                        <svg viewBox="0 0 120 120" width="80" height="80">...</svg>
                    </div>
                    <h3>福医卒中通助手</h3>
                </div>
                <div class="avatar-card">
                    <div class="avatar-box patient">
                        <svg viewBox="0 0 120 120" width="80" height="80">...</svg>
                    </div>
                    <h3 class="patient-text">咨询患者</h3>
                </div>
            </div>
            <div class="chat-main">
                <div class="quick-questions">
                    <button onclick="quickAsk('高血压怎么预防中风？')">高血压怎么预防中风？</button>
                    <button onclick="quickAsk('中风后吃什么好？')">中风后吃什么好？</button>
                    <button onclick="quickAsk('家人中风后怎么照顾？')">家人中风后怎么照顾？</button>
                    <button onclick="quickAsk('怎么判断是不是中风？')">怎么判断是不是中风？</button>
                    <button onclick="quickAsk('中风后手脚没力气怎么办？')">中风后手脚没力气怎么办？</button>
                    <button onclick="quickAsk('中风后情绪低落怎么办？')">中风后情绪低落怎么办？</button>
                    <button onclick="quickAsk('中风康复训练有哪些？')">中风康复训练有哪些？</button>
                    <button onclick="quickAsk('颈动脉斑块需要治疗吗？')">颈动脉斑块需要治疗吗？</button>
                    <button onclick="quickAsk('中风后可以运动吗？')">中风后可以运动吗？</button>
                    <button onclick="quickAsk('怎么帮家人做心理疏导？')">怎么帮家人做心理疏导？</button>
                </div>
                <div class="chat-body" id="chatBody">
                    <div class="message">
                        <div class="msg-avatar"><svg viewBox="0 0 44 44" width="26" height="26">...</svg></div>
                        <div class="msg-bubble">你好！我是脑卒中智能助手，可咨询预防、康复、养护、心理支持等问题~</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
<div class="chat-footer">
    <input class="chat-input" id="input" placeholder="问脑卒中相关问题：预防、康复、养护、心理支持...">
    <button class="mic-btn" id="micBtn">🎤</button>
    <button class="send-btn" id="sendBtn">发送</button>
    <button class="send-btn clear-btn" id="clearBtn">清空</button>
</div>
<div class="modal-mask" id="modalMask" onclick="closeFontModal()"></div>

<script>
let lang = "zh";
let voiceEnabled = true;
let fontOpt = "enlarge";
const synth = window.speechSynthesis;
let isRecording = false;
let activeRecognition = null;
let mediaStream = null;

const doctorAvatar = `<svg viewBox="0 0 44 44" width="26" height="26">
    <circle cx="22" cy="22" r="20" fill="#e6f7ff" stroke="#0077cc" stroke-width="1"/>
    <rect x="12" y="10" width="20" height="20" rx="3" fill="#f5d6c0" stroke="#333" stroke-width="1"/>
    <rect x="12" y="10" width="20" height="6" rx="1" fill="#2c3e50"/>
    <circle cx="17" cy="18" r="1.5" fill="#fff" stroke="#2c3e50" stroke-width="1"/>
    <circle cx="27" cy="18" r="1.5" fill="#fff" stroke="#2c3e50" stroke-width="1"/>
    <rect x="8" y="28" width="28" height="12" rx="2" fill="#fff" stroke="#0077cc" stroke-width="1"/>
</svg>`;
const patientAvatar = `<svg viewBox="0 0 44 44" width="26" height="26">
    <circle cx="22" cy="22" r="20" fill="#f0f9ff" stroke="#5499c7" stroke-width="1"/>
    <rect x="12" y="10" width="20" height="20" rx="3" fill="#f5d6c0" stroke="#333" stroke-width="1"/>
    <rect x="12" y="10" width="20" height="6" rx="1" fill="#2c3e50"/>
    <circle cx="17" cy="18" r="1.5" fill="#fff" stroke="#2c3e50" stroke-width="1"/>
    <circle cx="27" cy="18" r="1.5" fill="#fff" stroke="#2c3e50" stroke-width="1"/>
    <path d="M8 28 L12 26 L32 26 L36 28 L34 38 L10 38 Z" fill="#fff" stroke="#5499c7" stroke-width="1"/>
</svg>`;

var inputElement = document.getElementById("input");
if (inputElement) {
    inputElement.removeAttribute("readonly");
    inputElement.removeAttribute("disabled");
}

function selectOpt(opt) {
    fontOpt = opt;
    document.getElementById('enlargeBtn').className = opt === 'enlarge' ? 'opt-btn active' : 'opt-btn';
    document.getElementById('narrowBtn').className = opt === 'narrow' ? 'opt-btn active' : 'opt-btn';
    const scaleInput = document.getElementById('scaleInput');
    if (opt === 'enlarge') {
        scaleInput.min = 1;
        scaleInput.max = 4;
        scaleInput.step = 0.5;
        scaleInput.placeholder = "放大倍数（1-4）";
    } else {
        scaleInput.min = 0.3;
        scaleInput.max = 1;
        scaleInput.step = 0.1;
        scaleInput.placeholder = "缩小倍数（0.3-1）";
    }
    scaleInput.value = "";
    scaleInput.focus();
}

function adjustFont() {
    const scaleInput = document.getElementById('scaleInput');
    let rawValue = scaleInput.value.trim();
    let scale = parseFloat(rawValue);
    if (isNaN(scale) || scale <= 0) {
        scale = 1;
    }
    if (fontOpt === 'enlarge') {
        scale = Math.min(4, Math.max(1, scale));
    } else {
        scale = Math.min(1, Math.max(0.3, scale));
    }
    document.documentElement.style.setProperty('--font-scale', scale);
    scaleInput.value = scale;
    closeFontModal();
}

function openFontModal() {
    const modal = document.getElementById('fontModal');
    const mask = document.getElementById('modalMask');
    modal.classList.add('show');
    mask.classList.add('show');
    document.getElementById('scaleInput').focus();
}

function closeFontModal() {
    const modal = document.getElementById('fontModal');
    const mask = document.getElementById('modalMask');
    modal.classList.remove('show');
    mask.classList.remove('show');
    selectOpt('enlarge');
    document.getElementById('scaleInput').value = '';
}

function toggleVoice() {
    voiceEnabled = !voiceEnabled;
    const btn = document.getElementById("voiceBtn");
    btn.innerText = "语音播报：" + (voiceEnabled ? "开" : "关");
    if (!voiceEnabled) {
        if (synth) synth.cancel();
    } else {
        const lastMsg = getLastAssistantMessage();
        if (lastMsg) speak(lastMsg);
    }
}

function getLastAssistantMessage() {
    const messages = document.querySelectorAll('.message.assistant .msg-bubble');
    if (messages.length === 0) return null;
    const lastBubble = messages[messages.length - 1];
    return lastBubble.innerText.trim();
}

function speak(text) {
    if (!voiceEnabled) return;
    if (!text) return;
    if (synth) synth.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang === "zh" ? "zh-CN" : "en-US";
    utterance.onerror = function(e) {
        console.error("语音播报失败:", e);
    };
    synth.speak(utterance);
}

async function switchLang() {
    lang = lang === "zh" ? "en" : "zh";
    try {
        await fetch("/api/switch_lang", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lang })
        });
    } catch(e) { console.error("语言切换API调用失败", e); }
    document.getElementById("langBtn").innerText = lang === "zh" ? "切换英文" : "切换中文";
    clearChat();
}

async function ensureMicrophonePermission() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("您的浏览器不支持语音识别，请使用文字输入。");
        return false;
    }
    if (mediaStream && mediaStream.active) return true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaStream = stream;
        return true;
    } catch (err) {
        console.error("麦克风授权失败:", err);
        if (err.name === 'NotAllowedError') {
            alert("无法获取麦克风权限。请点击地址栏左侧锁图标 → 网站设置 → 麦克风 → 选择“允许”，然后刷新页面。");
        } else if (err.name === 'NotFoundError') {
            alert("未检测到麦克风设备，请检查耳机或麦克风连接。");
        } else {
            alert("麦克风授权申请失败: " + err.message);
        }
        return false;
    }
}

function startRecognition() {
    if (activeRecognition) {
        try { activeRecognition.abort(); } catch(e) {}
        activeRecognition = null;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;
    const recognition = new SpeechRecognition();
    recognition.lang = lang === "zh" ? "zh-CN" : "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => {
        isRecording = true;
        const btn = document.getElementById("micBtn");
        if (btn) btn.classList.add("recording");
    };
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        document.getElementById("input").value = transcript;
        if (activeRecognition) activeRecognition.stop();
    };
    recognition.onerror = (event) => {
        console.error("语音识别错误:", event.error);
        if (event.error === 'not-allowed') {
            alert("麦克风权限不足，请刷新页面后重新授权。");
        } else if (event.error !== 'aborted' && event.error !== 'no-speech') {
            alert(`语音识别出错: ${event.error}`);
        }
        stopRec();
    };
    recognition.onend = () => stopRec();
    try {
        recognition.start();
        activeRecognition = recognition;
    } catch (err) {
        console.error("启动语音识别异常:", err);
        alert("启动语音识别失败: " + err.message);
        stopRec();
    }
}

async function toggleRec() {
    if (isRecording) {
        stopRec();
        return;
    }
    const granted = await ensureMicrophonePermission();
    if (!granted) return;
    startRecognition();
}

function stopRec() {
    if (activeRecognition) {
        try { activeRecognition.abort(); } catch(e) {}
        activeRecognition = null;
    }
    isRecording = false;
    const btn = document.getElementById("micBtn");
    if (btn) btn.classList.remove("recording");
}

window.addEventListener('beforeunload', function() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (synth) synth.cancel();
});

async function send() {
    const inputEl = document.getElementById("input");
    const text = inputEl.value.trim();
    if (!text) return;
    addMsg("user", text);
    inputEl.value = "";
    
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "message assistant loading-message";
    loadingDiv.innerHTML = `<div class="msg-avatar">${doctorAvatar}</div><div class="msg-bubble">🤔 思考中...</div>`;
    const chatBody = document.getElementById("chatBody");
    chatBody.appendChild(loadingDiv);
    chatBody.scrollTop = chatBody.scrollHeight;
    
    try {
        const res = await fetch("/api/stroke_qa", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: text })
        });
        const data = await res.json();
        const ans = data.data.answer;
        loadingDiv.remove();
        addMsg("assistant", ans);
        speak(ans);
    } catch (err) {
        loadingDiv.remove();
        addMsg("assistant", "抱歉，网络错误，请稍后再试。");
        console.error(err);
    }
}

function addMsg(role, text) {
    const body = document.getElementById("chatBody");
    const div = document.createElement("div");
    div.className = "message " + role;
    let avatar = role === "user" ? patientAvatar : doctorAvatar;
    let cleanedText = text.replace(/\\*\\*/g, '');
    div.innerHTML = `<div class="msg-avatar">${avatar}</div><div class="msg-bubble">${cleanedText.replace(/\\n/g, '<br>')}</div>`;
    body.appendChild(div);
    body.scrollTop = body.scrollHeight;
}

function clearChat() {
    const body = document.getElementById("chatBody");
    body.innerHTML = `<div class="message"><div class="msg-avatar">${doctorAvatar}</div><div class="msg-bubble">${lang === "zh" ? "你好！我是脑卒中智能助手~" : "Hello! I'm stroke assistant~"}</div></div>`;
}

function quickAsk(question) {
    document.getElementById("input").value = question;
    send();
}

// 移动端适配（仅调整布局）
(function() {
    if (window.innerWidth <= 768) {
        function applyMobileStyles() {
            const sidebar = document.querySelector('.sidebar');
            const chatMain = document.querySelector('.chat-main');
            const chatContent = document.querySelector('.chat-content');
            if (sidebar) sidebar.style.display = 'none';
            if (chatMain) {
                chatMain.style.width = '100%';
                chatMain.style.flex = '1';
            }
            if (chatContent) {
                chatContent.style.display = 'flex';
                chatContent.style.flexDirection = 'row';
            }
            const style = document.createElement('style');
            style.textContent = `
                body header { margin-bottom: 8px !important; }
                body header p { display: none !important; }
                body .chat-header-bar { padding: 8px 12px !important; }
                body .chat-header-bar h2 { font-size: calc(20px * var(--font-scale)) !important; }
                body .header-btn { font-size: calc(12px * var(--font-scale)) !important; padding: 4px 8px !important; }
                body .chat-body { padding: 12px !important; }
                body .msg-bubble { padding: 8px 12px !important; }
                body .quick-questions button { font-size: calc(13px * var(--font-scale)) !important; padding: 6px 12px !important; }
                body .chat-footer { padding: 8px 12px !important; }
                body .chat-input { padding: 8px 12px !important; }
                body .send-btn, body .clear-btn { padding: 6px 12px !important; }
                body .mic-btn { width: 32px !important; height: 32px !important; }
                body .message { max-width: 90% !important; }
            `;
            document.head.appendChild(style);
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', applyMobileStyles);
        } else {
            applyMobileStyles();
        }
    }
})();

// 事件绑定
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById("fontBtn").onclick = openFontModal;
    document.getElementById("confirmFontBtn").onclick = adjustFont;
    document.getElementById("enlargeBtn").onclick = () => selectOpt('enlarge');
    document.getElementById("narrowBtn").onclick = () => selectOpt('narrow');
    document.getElementById("langBtn").onclick = switchLang;
    document.getElementById("voiceBtn").onclick = toggleVoice;
    document.getElementById("micBtn").onclick = toggleRec;
    document.getElementById("sendBtn").onclick = send;
    document.getElementById("clearBtn").onclick = clearChat;
    document.getElementById("input").onkeydown = function(e) {
        if (e.key === "Enter") {
            e.preventDefault();
            send();
        }
    };
    document.getElementById("modalMask").onclick = closeFontModal;
    document.getElementById("fontModal").onclick = function(e) {
        e.stopPropagation();
    };
    document.getElementById("scaleInput").onkeydown = function(e) {
        if (e.key === "Enter") adjustFont();
    };
    const inputEl = document.getElementById("input");
    if (inputEl) {
        inputEl.removeAttribute("readonly");
        inputEl.removeAttribute("disabled");
    }
});
</script>
</body>
</html>
''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5014, debug=False)
