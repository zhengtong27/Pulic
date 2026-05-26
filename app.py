# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template_string, Response
import re
import os
import json
import time
from dashscope import Application

app = Flask(__name__)

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
DASHSCOPE_APP_ID = os.environ.get("DASHSCOPE_APP_ID")
DASHSCOPE_WORKSPACE_ID = os.environ.get("DASHSCOPE_WORKSPACE_ID")

if not DASHSCOPE_API_KEY:
    print("警告：未设置环境变量 DASHSCOPE_API_KEY")
if not DASHSCOPE_APP_ID:
    print("警告：未设置环境变量 DASHSCOPE_APP_ID")

def generate_stream(question):
    mild = re.compile(
        r'(头(?:有?点)?痛|头(?:有?点)?晕|眼花|疲劳|乏力|失眠|焦虑|消化不良|颈部不适|有点不舒服)',
        re.IGNORECASE
    )
    if mild.search(question):
        fixed = ("头痛的原因很多，比如疲劳、紧张或血压波动。请先坐下休息，喝点温水，观察一下。"
                 "如果疼痛持续不缓解或加重，再咨询医生。注意：本内容仅供参考，如有需要请及时就医。")
        for i in range(0, len(fixed), 15):
            yield fixed[i:i+15]
            time.sleep(0.03)
        return
    try:
        resp = Application.call(
            app_id=DASHSCOPE_APP_ID,
            prompt=question,
            api_key=DASHSCOPE_API_KEY,
            workspace_id=DASHSCOPE_WORKSPACE_ID,
            stream=True,
            timeout=120
        )
    except Exception as e:
        print(e)
        yield "抱歉，系统繁忙，请稍后再试。"
        return
    last = ""
    buf = ""
    for chunk in resp:
        if chunk.output and chunk.output.text:
            txt = chunk.output.text
            if txt == last:
                continue
            last = txt
            buf += txt
            if len(buf) > 30 or buf.endswith(('。','！','？','\n')):
                yield buf
                buf = ""
    if buf:
        yield buf

@app.route('/api/stroke_qa_stream', methods=['POST'])
def stream_api():
    data = request.get_json() or {}
    q = data.get("question", "")
    if not q:
        return jsonify({"error": "问题为空"}), 400
    def gen():
        for chunk in generate_stream(q):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: {\"done\": true}\n\n"
    return Response(gen(), mimetype="text/event-stream")

@app.route('/api/switch_lang', methods=['POST'])
def switch_lang():
    return jsonify({"status": "success"})

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>福医卒中通 · 脑卒中智能诊疗助手</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        :root{--font-scale:1}
        body{background:linear-gradient(135deg,#f0f7ff,#e6f4fd);padding:20px;padding-bottom:90px}
        .container{max-width:1000px;margin:0 auto}
        header{text-align:center;margin-bottom:20px}
        header h1{font-size:calc(56px * var(--font-scale));color:#0077cc}
        header p{color:#666;font-size:calc(28px * var(--font-scale))}
        .main-card{background:#fff;border-radius:16px;box-shadow:0 8px 24px rgba(0,100,200,0.08);display:flex;flex-direction:column;height:calc(100vh - 40px - 90px)}
        .chat-header-bar{background:linear-gradient(90deg,#0077cc,#0099ee);color:#fff;padding:14px 20px;display:flex;justify-content:space-between}
        .chat-header-bar h2{font-size:calc(32px * var(--font-scale))}
        .header-btns{display:flex;gap:8px}
        .header-btn{padding:6px 12px;border-radius:20px;background:rgba(255,255,255,0.2);color:#fff;border:none;font-size:calc(26px * var(--font-scale));cursor:pointer}
        .chat-content{display:flex;flex:1;overflow:hidden}
        .sidebar{width:200px;background:#f8fcff;border-right:1px solid #e6f7ff;padding:20px}
        .avatar-box{width:100px;height:100px;margin:0 auto 10px;border-radius:50%;background:#fff;border:3px solid #0077cc;display:flex;align-items:center;justify-content:center;animation:breathing 4s infinite}
        .avatar-box.patient{border-color:#5499c7}
        .avatar-card{text-align:center;margin-bottom:24px}
        .avatar-card h3{font-size:calc(28px * var(--font-scale));color:#0077cc}
        @keyframes breathing{0%,100%{transform:scale(1)}50%{transform:scale(1.04)}}
        .chat-main{flex:1;display:flex;flex-direction:column;overflow:hidden}
        .quick-questions{padding:12px 16px;background:#f5f9fe;border-bottom:1px solid #e6f0fa;overflow-x:auto;white-space:nowrap;display:flex;gap:10px}
        .quick-questions button{background:#eef3fc;border:1px solid #cce4f5;border-radius:24px;padding:8px 16px;font-size:calc(24px * var(--font-scale));color:#0077cc;cursor:pointer}
        .chat-body{flex:1;padding:20px;overflow-y:auto;background:#fafbfc}
        .message{display:flex;gap:10px;margin-bottom:14px;max-width:75%}
        .message.user{margin-left:auto;flex-direction:row-reverse}
        .msg-avatar{width:36px;height:36px;border-radius:50%;background:#fff;border:1px solid #eee;display:flex;align-items:center;justify-content:center}
        .msg-bubble{padding:10px 14px;border-radius:14px;background:#fff;border:1px solid #eee;line-height:1.5;font-size:calc(28px * var(--font-scale))}
        .message.user .msg-bubble{background:#0077cc;color:#fff;border:none}
        .chat-footer{position:fixed;bottom:0;left:0;width:100%;padding:14px 20px;border-top:1px solid #eee;display:flex;gap:10px;background:#fff;z-index:99}
        .chat-input{flex:1;padding:10px 16px;border:1px solid #ddd;border-radius:24px;outline:none;font-size:calc(28px * var(--font-scale))}
        .send-btn,.clear-btn{padding:10px 18px;border-radius:24px;background:#0077cc;color:#fff;border:none;font-size:calc(28px * var(--font-scale));cursor:pointer}
        .clear-btn{background:#777}
        .mic-btn{width:38px;height:38px;border-radius:50%;background:#0077cc;color:#fff;font-size:calc(32px * var(--font-scale));border:none;cursor:pointer}
        .mic-btn.recording{background:#e53935;animation:pulse 1s infinite}
        @keyframes pulse{0%{transform:scale(1)}50%{transform:scale(1.1)}100%{transform:scale(1)}}
        .modal-mask{position:fixed;top:0;left:0;width:100%;height:100%;z-index:999;background:rgba(0,0,0,0.5);display:none}
        .font-modal{position:absolute;top:60px;right:0;background:#fff;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.1);padding:15px;z-index:1000;display:none;min-width:280px}
        .font-modal.show{display:block}
        @media (max-width:768px){.sidebar{display:none}.chat-main{width:100%}}
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
                        <input type="number" id="scaleInput" step="0.5" value="">
                    </div>
                    <button class="confirm-btn" id="confirmFontBtn">确认调节</button>
                    <div class="tip-text">放大1-4 缩小0.3-1</div>
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
let lang = "zh", voiceEnabled = true, fontOpt = "enlarge", isRecording = false, activeRecognition = null, mediaStream = null;
const synth = window.speechSynthesis;
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

if(document.getElementById("input")) document.getElementById("input").removeAttribute("readonly");
if(document.getElementById("input")) document.getElementById("input").removeAttribute("disabled");

function selectOpt(opt) {
    fontOpt = opt;
    let enlargeBtn = document.getElementById("enlargeBtn");
    let narrowBtn = document.getElementById("narrowBtn");
    if(enlargeBtn) enlargeBtn.className = opt==='enlarge'?'opt-btn active':'opt-btn';
    if(narrowBtn) narrowBtn.className = opt==='narrow'?'opt-btn active':'opt-btn';
    let si = document.getElementById("scaleInput");
    if(si) {
        if(opt==='enlarge'){ si.min=1; si.max=4; si.step=0.5; si.placeholder="1-4"; }
        else{ si.min=0.3; si.max=1; si.step=0.1; si.placeholder="0.3-1"; }
        si.value=""; si.focus();
    }
}
function adjustFont() {
    let si = document.getElementById("scaleInput");
    if(!si) return;
    let v = parseFloat(si.value.trim());
    if(isNaN(v)||v<=0) v=1;
    if(fontOpt==='enlarge') v=Math.min(4,Math.max(1,v));
    else v=Math.min(1,Math.max(0.3,v));
    document.documentElement.style.setProperty('--font-scale', v);
    si.value=v;
    closeFontModal();
}
function openFontModal(){
    let modal = document.getElementById("fontModal");
    let mask = document.getElementById("modalMask");
    if(modal) modal.classList.add('show');
    if(mask) mask.classList.add('show');
    let input = document.getElementById("scaleInput");
    if(input) input.focus();
}
function closeFontModal(){
    let modal = document.getElementById("fontModal");
    let mask = document.getElementById("modalMask");
    if(modal) modal.classList.remove('show');
    if(mask) mask.classList.remove('show');
    selectOpt('enlarge');
    let input = document.getElementById("scaleInput");
    if(input) input.value = '';
}

async function ensureMic() {
    if(!window.SpeechRecognition && !window.webkitSpeechRecognition){ alert("不支持语音识别"); return false; }
    if(mediaStream && mediaStream.active) return true;
    try{ const s = await navigator.mediaDevices.getUserMedia({audio:true}); mediaStream=s; return true; }
    catch(e){ alert("请允许麦克风权限"); return false; }
}
function startRecog(){
    if(activeRecognition) activeRecognition.abort();
    const Rec = window.SpeechRecognition||window.webkitSpeechRecognition;
    if(!Rec) return;
    const rec = new Rec();
    rec.lang = lang==='zh'?'zh-CN':'en-US';
    rec.interimResults=false;
    rec.maxAlternatives=1;
    rec.onstart=()=>{ isRecording=true; let btn = document.getElementById("micBtn"); if(btn) btn.classList.add('recording'); };
    rec.onresult=(e)=>{ let input = document.getElementById("input"); if(input) input.value=e.results[0][0].transcript; rec.stop(); };
    rec.onerror=()=>stopRec();
    rec.onend=()=>stopRec();
    try{ rec.start(); activeRecognition=rec; }catch(e){ alert("启动失败"); stopRec(); }
}
async function toggleRec(){
    if(isRecording){ stopRec(); return; }
    if(await ensureMic()) startRecog();
}
function stopRec(){
    if(activeRecognition) activeRecognition.abort();
    activeRecognition=null;
    isRecording=false;
    let btn = document.getElementById("micBtn");
    if(btn) btn.classList.remove('recording');
}
window.addEventListener('beforeunload',()=>{ if(mediaStream) mediaStream.getTracks().forEach(t=>t.stop()); if(synth) synth.cancel(); });

function toggleVoice(){
    voiceEnabled=!voiceEnabled;
    let btn = document.getElementById("voiceBtn");
    if(btn) btn.innerText="语音播报："+(voiceEnabled?"开":"关");
    if(!voiceEnabled){ if(synth) synth.cancel(); }
    else{ let last = document.querySelector('.message.assistant .msg-bubble:last-child'); if(last) speak(last.innerText); }
}
function getLastAssistantMessage(){ let msgs = document.querySelectorAll('.message.assistant .msg-bubble'); if(msgs.length) return msgs[msgs.length-1].innerText.trim(); return null; }
function speak(t){ if(!voiceEnabled||!t) return; if(synth) synth.cancel(); let u=new SpeechSynthesisUtterance(t); u.lang=lang==='zh'?'zh-CN':'en-US'; if(synth) synth.speak(u); }
async function switchLang(){
    lang=lang==='zh'?'en':'zh';
    await fetch('/api/switch_lang',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lang})}).catch(e=>console.log);
    let btn = document.getElementById("langBtn");
    if(btn) btn.innerText=lang==='zh'?'切换英文':'切换中文';
    clearChat();
}
function addMsg(role, text){
    let body=document.getElementById('chatBody');
    if(!body) return;
    let div=document.createElement('div');
    div.className='message '+role;
    let avatar=role==='user'?patientAvatar:doctorAvatar;
    let clean=text.replace(/\\*\\*/g,'');
    div.innerHTML=`<div class="msg-avatar">${avatar}</div><div class="msg-bubble">${clean.replace(/\\n/g,'<br>')}</div>`;
    body.appendChild(div);
    body.scrollTop=body.scrollHeight;
}
function clearChat(){
    let body=document.getElementById('chatBody');
    if(body) body.innerHTML=`<div class="message"><div class="msg-avatar">${doctorAvatar}</div><div class="msg-bubble">${lang==='zh'?'你好！我是脑卒中智能助手~':'Hello! I'm stroke assistant~'}</div></div>`;
}
async function send(){
    let input = document.getElementById("input");
    if(!input) return;
    let text = input.value.trim();
    if(!text) return;
    addMsg('user',text);
    input.value='';
    let loading=document.createElement('div');
    loading.className='message assistant loading-message';
    loading.innerHTML=`<div class="msg-avatar">${doctorAvatar}</div><div class="msg-bubble">🤔 思考中...</div>`;
    let chatBody=document.getElementById('chatBody');
    if(chatBody) chatBody.appendChild(loading);
    if(chatBody) chatBody.scrollTop=chatBody.scrollHeight;
    try{
        let resp=await fetch('/api/stroke_qa_stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text})});
        if(loading) loading.remove();
        let assistantDiv=document.createElement('div');
        assistantDiv.className='message assistant';
        assistantDiv.innerHTML=`<div class="msg-avatar">${doctorAvatar}</div><div class="msg-bubble"></div>`;
        if(chatBody) chatBody.appendChild(assistantDiv);
        let bubble=assistantDiv.querySelector('.msg-bubble');
        let full='';
        let reader=resp.body.getReader();
        let decoder=new TextDecoder();
        let buffer='';
        while(true){
            let {done,value}=await reader.read();
            if(done) break;
            buffer+=decoder.decode(value,{stream:true});
            let lines=buffer.split('\n\n');
            buffer=lines.pop();
            for(let line of lines){
                if(line.startsWith('data: ')){
                    let json=line.slice(6);
                    try{
                        let data=JSON.parse(json);
                        if(data.chunk){
                            full+=data.chunk;
                            if(bubble) bubble.innerHTML=full.replace(/\\n/g,'<br>');
                            if(chatBody) chatBody.scrollTop=chatBody.scrollHeight;
                        }else if(data.done){}
                    }catch(e){}
                }
            }
        }
        if(voiceEnabled && full) speak(full);
    }catch(e){
        if(loading) loading.remove();
        addMsg('assistant','抱歉，网络错误，请稍后再试。');
        console.error(e);
    }
}
function quickAsk(q){ let input=document.getElementById("input"); if(input) input.value=q; send(); }

document.addEventListener('DOMContentLoaded',()=>{
    let fontBtn = document.getElementById("fontBtn");
    if(fontBtn) fontBtn.onclick = openFontModal;
    let confirmBtn = document.getElementById("confirmFontBtn");
    if(confirmBtn) confirmBtn.onclick = adjustFont;
    let enlargeBtn = document.getElementById("enlargeBtn");
    if(enlargeBtn) enlargeBtn.onclick = ()=>selectOpt('enlarge');
    let narrowBtn = document.getElementById("narrowBtn");
    if(narrowBtn) narrowBtn.onclick = ()=>selectOpt('narrow');
    let langBtn = document.getElementById("langBtn");
    if(langBtn) langBtn.onclick = switchLang;
    let voiceBtn = document.getElementById("voiceBtn");
    if(voiceBtn) voiceBtn.onclick = toggleVoice;
    let micBtn = document.getElementById("micBtn");
    if(micBtn) micBtn.onclick = toggleRec;
    let sendBtn = document.getElementById("sendBtn");
    if(sendBtn) sendBtn.onclick = send;
    let clearBtn = document.getElementById("clearBtn");
    if(clearBtn) clearBtn.onclick = clearChat;
    let input = document.getElementById("input");
    if(input) input.onkeydown = e => { if(e.key==='Enter'){ e.preventDefault(); send(); } };
    let modalMask = document.getElementById("modalMask");
    if(modalMask) modalMask.onclick = closeFontModal;
    let fontModal = document.getElementById("fontModal");
    if(fontModal) fontModal.onclick = e => e.stopPropagation();
    let scaleInput = document.getElementById("scaleInput");
    if(scaleInput) scaleInput.onkeydown = e => { if(e.key==='Enter') adjustFont(); };
    if(input) input.removeAttribute("readonly");
    if(input) input.removeAttribute("disabled");
});
</script>
</body>
</html>
''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5014, debug=False)
