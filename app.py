# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template, Response
import re
import os
import json
import time
from openai import OpenAI
from dashscope import Retrieval

app = Flask(__name__)

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("警告：未设置环境变量 DASHSCOPE_API_KEY")

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

if DASHSCOPE_API_KEY:
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
else:
    client = None

MODEL_NAME = "qwen3-32b-351ed038aecc"
KNOWLEDGE_BASE_ID = "ssy8053dlh"   # 替换为你的百炼知识库ID

def retrieve_context(question):
    """从百炼知识库检索相关内容"""
    try:
        resp = Retrieval.search(
            index_name=KNOWLEDGE_BASE_ID,
            query=question,
            dense_similarity_top_k=3,
            sparse_similarity_top_k=3,
            enable_reranking=True,
            rerank_top_n=3
        )
        docs = []
        if resp and hasattr(resp, 'output') and resp.output:
            for doc in resp.output.documents:
                docs.append(doc.text)
        return "\n\n".join(docs)
    except Exception as e:
        print("检索失败:", e)
        return ""

def generate_stream(question):
    # 非紧急症状过滤
    mild_pattern = re.compile(
        r'(头(?:有?点)?痛|头(?:有?点)?晕|眼花|疲劳|乏力|失眠|焦虑|消化不良|颈部不适|有点不舒服)',
        re.IGNORECASE
    )
    if mild_pattern.search(question):
        fixed = ("头痛的原因很多，比如疲劳、紧张或血压波动。请先坐下休息，喝点温水，观察一下。"
                 "如果疼痛持续不缓解或加重，再咨询医生。注意：本内容仅供参考，如有需要请及时就医。")
        for ch in fixed:
            yield ch
            time.sleep(0.03)
        return

    # 检索知识库
    context = retrieve_context(question)
    
    system_prompt = (
        "你是一个脑卒中健康科普助手，专为老年人及家属提供温和、可信的健康知识。\n\n"
        "【回答风格】\n"
        "直接回答用户的问题，不要以“您说得对”、“好的”、“是的”等肯定性词语开头。保持语气温和、简洁。\n\n"
    )
    if context:
        system_prompt += f"【参考资料】\n{context}\n\n请优先使用上述参考资料回答用户问题。\n\n"
    system_prompt += (
        "【重要限制】\n"
        "1. 对于以下症状，绝对不要提及“脑卒中”、“中风”、“紧急就医”、“拨打120”等词汇，只需给予休息观察建议：\n"
        "   - 轻微头痛、头晕、眼花、疲劳、乏力、颈部不适、失眠、焦虑、消化不良等\n"
        "2. 只有当用户明确描述以下至少一项脑卒中典型征兆时，才明确建议立即就医：\n"
        "   - 一侧肢体突然无力或麻木\n"
        "   - 口角歪斜、说话不清\n"
        "   - 突发剧烈头痛（“像被雷劈一样”）\n"
        "   - 单侧视力突然模糊或失明\n"
        "   - 突然行走不稳、失去平衡\n"
        "3. 对于所有其他健康问题，回答应通俗易懂，始终强调“本内容仅供参考，如有不适请及时就医”。\n"
        "4. 绝不提供急救指导、药物剂量或替代医生诊断的建议。\n\n"
        "【来源要求】\n"
        "在回答末尾附上主要参考来源，例如（来源：《中国脑卒中防治指南2023》）。"
    )
    
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            top_p=0.85,
            max_tokens=1024,
            stream=True
        )
    except Exception as e:
        print(e)
        err = "抱歉，系统繁忙，请稍后再试。"
        for ch in err:
            yield ch
            time.sleep(0.03)
        return
    
    for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                yield delta.content   # 直接输出完整块，不做逐字符拆分，避免重复

@app.route('/api/stream', methods=['POST'])
def stream():
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
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5014, debug=False)
