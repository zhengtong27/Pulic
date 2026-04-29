# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template_string
import re
import os
from openai import OpenAI

app = Flask(__name__)

# 获取 API Key
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("警告：未设置环境变量 DASHSCOPE_API_KEY，API 调用将失败")

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if DASHSCOPE_API_KEY:
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
else:
    client = None
    print("错误：无法创建 OpenAI 客户端，请设置环境变量 DASHSCOPE_API_KEY")

MODEL_NAME = "qwen3-32b-27649d93fc36"   # 请根据实际情况修改

@app.after_request
def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Cache-Control'] = 'no-cache'
    return response

def call_llm(question):
    mild_pattern = re.compile(
        r'(头(?:有?点)?痛|头(?:有?点)?晕|眼花|疲劳|乏力|失眠|焦虑|消化不良|颈部不适|有点不舒服)',
        re.IGNORECASE
    )
    if mild_pattern.search(question):
        return ("头痛的原因很多，比如疲劳、紧张或血压波动。请先坐下休息，喝点温水，观察一下。"
                "如果疼痛持续不缓解或加重，再咨询医生。注意：本内容仅供参考，如有需要请及时就医。")
    
    system_prompt = (
        "你是一个脑卒中健康科普助手，专为老年人及家属提供温和、可信的健康知识。\n\n"
        "【回答风格】\n"
        "直接回答用户的问题，不要以“您说得对”、“好的”、“是的”等肯定性词语开头。保持语气温和、简洁，直接给出建议或信息。\n\n"
        "【重要限制】\n"
        "1. 对于以下症状，绝对不要提及“脑卒中”、“中风”、“紧急就医”、“拨打120”等词汇，只需给予休息观察建议：\n"
        "   - 轻微头痛、头晕、眼花、疲劳、乏力、颈部不适、失眠、焦虑、消化不良等\n"
        "   - 回答示例：\n"
        "     “头痛的原因很多，比如疲劳、紧张或血压波动。请先坐下休息，喝点温水，观察一下。如果疼痛持续不缓解或加重，再咨询医生。”\n\n"
        "2. 只有当用户明确描述以下至少一项脑卒中典型征兆时，才明确建议立即就医：\n"
        "   - 一侧肢体突然无力或麻木\n"
        "   - 口角歪斜、说话不清\n"
        "   - 突发剧烈头痛（“像被雷劈一样”）\n"
        "   - 单侧视力突然模糊或失明\n"
        "   - 突然行走不稳、失去平衡\n\n"
        "3. 对于所有其他健康问题，回答应通俗易懂，引用权威知识，但始终强调“本内容仅供参考，如有不适请及时就医”。\n\n"
        "4. 绝不提供急救指导、药物剂量或替代医生诊断的建议。\n\n"
        "5. 如果用户描述的症状不在上述列表中，请先询问是否有其他症状，并建议先休息观察，切勿自行套用脑卒中标准。"
    )
    
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            extra_body={"enable_thinking": True},
            temperature=0.3,
            top_p=0.85,
            max_tokens=1024,
            stream=True
        )
        
        full_answer = ""
        reasoning_parts = []
        
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_parts.append(delta.reasoning_content)
                if hasattr(delta, "content") and delta.content:
                    full_answer += delta.content
        
        if reasoning_parts:
            print(f"[思考过程] {''.join(reasoning_parts)}")
        
        prefix_pattern = re.compile(r'^(您说得对|好的|是的|没错|嗯|对，|对的，|好的，)\s*', re.IGNORECASE)
        cleaned_answer = prefix_pattern.sub('', full_answer).strip()
        if cleaned_answer:
            full_answer = cleaned_answer
        
        emergency_keywords = ["脑卒中", "中风", "拨打120", "紧急就医", "立即前往医院", "专业医生进行评估"]
        if full_answer and any(kw in full_answer for kw in emergency_keywords):
            if mild_pattern.search(question):
                return ("头痛的原因很多，比如疲劳、紧张或血压波动。请先坐下休息，喝点温水，观察一下。"
                        "如果疼痛持续不缓解或加重，再咨询医生。注意：本内容仅供参考，如有需要请及时就医。")
        
        return full_answer.strip() if full_answer else "抱歉，模型未返回有效回答。"
    
    except Exception as e:
        print(f"模型调用失败: {e}")
        return "抱歉，系统繁忙，请稍后再试。"

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
    return render_template_string('''
<!DOCTYPE html>
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
            z-index: 100;
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
            z-index: 98;
            display: none;
        }
        .modal-mask.show {
            display: block;
        }

        /* ========== 移动端适配 (强制生效) ========== */
        @media (max-width: 768px) {
            .sidebar {
                display: none !important;
            }
            .chat-main {
                width: 100% !important;
            }
            .container {
                padding: 0 8px !important;
            }
            body {
                padding: 10px 0 90px 0 !important;
            }
            .main-card {
                height: calc(100vh - 20px - 80px) !important;
                border-radius: 12px !important;
            }
            .message {
                max-width: 90% !important;
            }
            .msg-bubble {
                font-size: calc(32px * var(--font-scale)) !important;
                padding: 10px 14px !important;
            }
            .quick-questions {
                padding: 10px 12px !important;
                gap: 10px !important;
                overflow-x: auto !important;
            }
            .quick-questions button {
                padding: 12px 18px !important;
                font-size: calc(28px * var(--font-scale)) !important;
                flex-shrink: 0 !important;
            }
            .chat-footer {
                padding: 10px 12px !important;
                gap: 12px !important;
                flex-wrap: nowrap !important;
                padding-bottom: calc(12px + env(safe-area-inset-bottom)) !important;
            }
            .chat-input {
                font-size: calc(32px * var(--font-scale)) !important;
                padding: 12px 14px !important;
            }
            .send-btn, .clear-btn {
                font-size: calc(30px * var(--font-scale)) !important;
                padding: 10px 18px !important;
            }
            .mic-btn {
                width: 48px !important;
                height: 48px !important;
                font-size: 28px !important;
            }
            .header-btn {
                padding: 8px 12px !important;
                font-size: calc(26px * var(--font-scale)) !important;
            }
            header h1 {
                font-size: calc(44px * var(--font-scale)) !important;
                line-height: 1.2 !important;
            }
            header p {
                font-size: calc(24px * var(--font-scale)) !important;
                padding: 0 16px !important;
            }
            .chat-header-bar h2 {
                font-size: calc(34px * var(--font-scale)) !important;
            }
            .chat-body {
                padding: 16px 12px !important;
            }
        }

        /* 特别小的手机 */
        @media (max-width: 480px) {
            .quick-questions button {
                padding: 10px 16px !important;
                font-size: calc(26px * var(--font-scale)) !important;
            }
            .send-btn, .clear-btn {
                padding: 8px 14px !important;
                font-size: calc(28px * var(--font-scale)) !important;
            }
            .mic-btn {
                width: 44px !important;
                height: 44px !important;
                font-size: 26px !important;
            }
            .chat-input {
                font-size: calc(30px * var(--font-scale)) !important;
                padding: 10px 12px !important;
            }
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
                        <button class="opt-btn active" id="enlargeBtn" onclick="selectOpt('enlarge')">放大</button>
                        <button class="opt-btn" id="narrowBtn" onclick="selectOpt('narrow')">缩小</button>
                    </div>
                    <div class="input-group">
                        <input type="number" id="scaleInput" placeholder="请输入调节倍数" min="1" max="4" step="0.5" value="1">
                    </div>
                    <button class="confirm-btn" onclick="adjustFont()">确认调节</button>
                    <div class="tip-text">放大：1-4（步长0.5）| 缩小：0.3-1（步长0.1）</div>
                </div>
            </div>
        </div>
        <div class="chat-content">
            <div class="sidebar">
                <div class="avatar-card">
                    <div class="avatar-box">
                        <svg viewBox="0 0 120 120" width="80" height="80">
                            <circle cx="60" cy="60" r="58" fill="#e6f7ff" stroke="#0077cc" stroke-width="1"/>
                            <rect x="35" y="25" width="50" height="50" rx="8" fill="#f5d6c0" stroke="#333" stroke-width="1.5"/>
                            <rect x="35" y="25" width="50" height="12" rx="2" fill="#2c3e50"/>
                            <rect x="32" y="22" width="10" height="8" rx="1" fill="#2c3e50"/>
                            <rect x="78" y="22" width="10" height="8" rx="1" fill="#2c3e50"/>
                            <circle cx="50" cy="50" r="3" fill="#fff" stroke="#2c3e50" stroke-width="1.5"/>
                            <circle cx="70" cy="50" r="3" fill="#fff" stroke="#2c3e50" stroke-width="1.5"/>
                            <line x1="50" y1="65" x2="70" y2="65" stroke="#2c3e50" stroke-width="1.5" stroke-linecap="round"/>
                            <rect x="25" y="75" width="70" height="35" rx="4" fill="#ffffff" stroke="#0077cc" stroke-width="2"/>
                            <line x1="60" y1="75" x2="60" y2="110" stroke="#0077cc" stroke-width="1.5"/>
                            <circle cx="40" cy="90" r="4" fill="#0077cc" stroke="#333" stroke-width="1"/>
                            <circle cx="80" cy="90" r="4" fill="#0077cc" stroke="#333" stroke-width="1"/>
                            <path d="M40 90 C30 80, 90 80, 80 90" stroke="#0077cc" stroke-width="2" fill="none"/>
                        </svg>
                    </div>
                    <h3>福医卒中通助手</h3>
                </div>
                <div class="avatar-card">
                    <div class="avatar-box patient">
                        <svg viewBox="0 0 120 120" width="80" height="80">
                            <circle cx="60" cy="60" r="58" fill="#f0f9ff" stroke="#5499c7" stroke-width="1"/>
                            <rect x="35" y="25" width="50" height="50" rx="8" fill="#f5d6c0" stroke="#333" stroke-width="1.5"/>
                            <rect x="35" y="25" width="50" height="12" rx="2" fill="#2c3e50"/>
                            <rect x="32" y="22" width="8" height="7" rx="1" fill="#2c3e50"/>
                            <rect x="80" y="22" width="8" height="7" rx="1" fill="#2c3e50"/>
                            <circle cx="50" cy="50" r="3" fill="#fff" stroke="#2c3e50" stroke-width="1.5"/>
                            <circle cx="70" cy="50" r="3" fill="#fff" stroke="#2c3e50" stroke-width="1.5"/>
                            <line x1="50" y1="65" x2="70" y2="65" stroke="#2c3e50" stroke-width="1.5" stroke-linecap="round"/>
                            <path d="M25 75 L35 70 L85 70 L95 75 L90 110 L30 110 Z" fill="#ffffff" stroke="#5499c7" stroke-width="2"/>
                            <line x1="60" y1="70" x2="60" y2="110" stroke="#5499c7" stroke-width="1.5"/>
                            <rect x="50" y="70" width="20" height="5" rx="2" fill="#f0f9ff" stroke="#5499c7" stroke-width="1.5"/>
                        </svg>
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
                        <div class="msg-avatar">
                            <svg viewBox="0 0 44 44" width="26" height="26">
                                <circle cx="22" cy="22" r="20" fill="#e6f7ff" stroke="#0077cc" stroke-width="1"/>
                                <rect x="12" y="10" width="20" height="20" rx="3" fill="#f5d6c0" stroke="#333" stroke-width="1"/>
                                <rect x="12" y="10" width="20" height="6" rx="1" fill="#2c3e50"/>
                                <circle cx="17" cy="18" r="1.5" fill="#fff" stroke="#2c3e50" stroke-width="1"/>
                                <circle cx="27" cy="18" r="1.5" fill="#fff" stroke="#2c3e50" stroke-width="1"/>
                                <rect x="8" y="28" width="28" height="12" rx="2" fill="#fff" stroke="#0077cc" stroke-width="1"/>
                            </svg>
                        </div>
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
let recognition = null;
let isRecording = false;

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
        scaleInput.value = 1;
        scaleInput.placeholder = "放大倍数（1-4）";
    } else {
        scaleInput.min = 0.3;
        scaleInput.max = 1;
        scaleInput.step = 0.1;
        scaleInput.value = 1;
        scaleInput.placeholder = "缩小倍数（0.3-1）";
    }
    scaleInput.focus();
}

function adjustFont() {
    const scaleInput = document.getElementById('scaleInput');
    const inputVal = scaleInput.value.trim();
    if (!inputVal || isNaN(inputVal) || Number(inputVal) <= 0) {
        alert("请输入有效的正数倍数！");
        scaleInput.focus();
        return;
    }
    const scale = Number(inputVal);
    document.documentElement.style.setProperty('--font-scale', scale);
    closeFontModal();
    scaleInput.value = '';
}

function openFontModal() {
    document.getElementById('fontModal').classList.add('show');
    document.getElementById('modalMask').classList.add('show');
    document.getElementById('scaleInput').focus();
}

function closeFontModal() {
    document.getElementById('fontModal').classList.remove('show');
    document.getElementById('modalMask').classList.remove('show');
    selectOpt('enlarge');
    document.getElementById('scaleInput').value = '1';
}

function initRecognition() {
    if (!window.SpeechRecognition && !window.webkitSpeechRecognition) {
        alert("当前浏览器不支持语音输入，请使用 Chrome 或 Edge 最新版。");
        return false;
    }
    const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new Rec();
    recognition.lang = lang === "zh" ? "zh-CN" : "en-US";
    recognition.interimResults = false;
    recognition.continuous = false;
    recognition.onresult = (e) => {
        const txt = e.results[0][0].transcript;
        document.getElementById("input").value = txt;
        stopRec();
    };
    recognition.onerror = stopRec;
    recognition.onend = stopRec;
    return true;
}

function toggleRec() {
    if (!recognition) {
        if (!initRecognition()) return;
    }
    isRecording = !isRecording;
    const btn = document.getElementById("micBtn");
    btn.classList.toggle("recording");
    if (isRecording) {
        recognition.start();
    } else {
        recognition.stop();
    }
}

function stopRec() {
    isRecording = false;
    document.getElementById("micBtn").classList.remove("recording");
    if (recognition) recognition.stop();
}

function toggleVoice() {
    voiceEnabled = !voiceEnabled;
    document.getElementById("voiceBtn").innerText = "语音播报：" + (voiceEnabled ? "开" : "关");
    if (!voiceEnabled) {
        synth.cancel();
    } else {
        const lastMsg = getLastAssistantMessage();
        if (lastMsg) speak(lastMsg);
    }
}

function getLastAssistantMessage() {
    const messages = document.querySelectorAll('.message.assistant .msg-bubble');
    if (messages.length === 0) return null;
    const lastBubble = messages[messages.length - 1];
    let text = lastBubble.innerText || lastBubble.textContent;
    return text.trim();
}

function speak(text) {
    if (!voiceEnabled) return;
    synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang === "zh" ? "zh-CN" : "en-US";
    synth.speak(u);
}

async function send() {
    const text = document.getElementById("input").value.trim();
    if (!text) return;
    addMsg("user", text);
    document.getElementById("input").value = "";
    
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

async function switchLang() {
    lang = lang === "zh" ? "en" : "zh";
    await fetch("/api/switch_lang", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lang })
    });
    document.getElementById("langBtn").innerText = lang === "zh" ? "切换英文" : "切换中文";
    clearChat();
}

function clearChat() {
    const body = document.getElementById("chatBody");
    body.innerHTML = `<div class="message"><div class="msg-avatar">${doctorAvatar}</div><div class="msg-bubble">${lang === "zh" ? "你好！我是脑卒中智能助手~" : "Hello! I'm stroke assistant~"}</div></div>`;
}

function quickAsk(question) {
    const input = document.getElementById("input");
    input.value = question;
    send();
}

document.getElementById("sendBtn").onclick = send;
document.getElementById("clearBtn").onclick = clearChat;
document.getElementById("langBtn").onclick = switchLang;
document.getElementById("micBtn").onclick = toggleRec;
document.getElementById("voiceBtn").onclick = toggleVoice;
document.getElementById("fontBtn").onclick = openFontModal;

document.getElementById("input").onkeydown = function(e) {
    if (e.key === "Enter") {
        e.preventDefault();
        send();
    }
};

document.getElementById("fontModal").onclick = e => e.stopPropagation();
document.getElementById("scaleInput").onkeydown = e => e.key === "Enter" && adjustFont();
</script>
</body>
</html>
''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5014, debug=False)
