import os, glob, pickle, readline
import sentencepiece as spm
import pandas as pd
import torch, faiss
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

#  1) 读取所有文档
docs_dir = "docs"  # 文档文件夹
files = glob.glob(os.path.join(docs_dir, "**/*.*"), recursive=True)
raw_texts = []

for f in files:
    suf = f.lower()
    try:
        if suf.endswith(".pdf"):# pdf 格式
            raw_texts += [p.page_content for p in PyPDFLoader(f).load()]

        elif suf.endswith(".txt"): #文本格式
            raw_texts += TextLoader(f, encoding="utf-8").load()

        elif suf.endswith(".xlsx"): # xlsx 格式
            df_dict = pd.read_excel(f, sheet_name=None, header=None,
                                    engine="openpyxl")
            for df in df_dict.values():
                txt = (df.astype(str).fillna("")
                       .agg(" ".join, axis=1).str.cat(sep="\n"))
                raw_texts.append(txt)

        elif suf.endswith(".xls"): # xls格式
            df_dict = pd.read_excel(f, sheet_name=None, header=None,
                                    engine="xlrd")
            for df in df_dict.values():
                txt = (df.astype(str).fillna("")
                       .agg(" ".join, axis=1).str.cat(sep="\n"))
                raw_texts.append(txt)

    except Exception as e:
        print(f"读取失败 {f}: {e}")

print(f"已收集 {len(raw_texts)} 条原始文本")

#  2) 数据切片
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = []
for doc in raw_texts:
    chunks += splitter.split_text(doc.page_content if hasattr(doc, "page_content") else doc)

print(f" 已生成 {len(chunks)} 个文本块")

#  3) 向量化
emb_model = SentenceTransformer(
    r"C:\models\bge-base-zh-v1.5",          # ← 本地的bge模型目录
    trust_remote_code=True)

embeddings = emb_model.encode(chunks, batch_size=64, show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
with open("faiss_store.pkl", "wb") as fp:
    pickle.dump((index, chunks), fp)

print("向量库保存至 faiss_store.pkl")

#  4) 加载 Baichuan‑13B GPTQ
model_path = r"C:\RAG_demo\baichuan-13b-chat"  # 权重目录
offload_dir = r"C:\RAG_demo\offload"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

if not hasattr(tok, "sp_model"):
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(model_path, "spm.model"))
    tok.sp_model = sp
    tok.vocab_size = sp.get_piece_size()

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    offload_folder=offload_dir,
    torch_dtype=dtype,
    trust_remote_code=True,
    low_cpu_mem_usage=True)

print("模型加载完成，输入空行退出")

def build_prompt(question: str, context: str) -> str:
    if hasattr(tok, "apply_chat_template"):
        messages = [
            {"role": "system",
             "content": "你是洁净室工程知识助手，请基于已知信息准确回答。"},
            {"role": "user",
             "content": f"已知信息：\n{context}\n\n问题：{question}"}
        ]
        return tok.apply_chat_template(messages,
                                       add_generation_prompt=True,
                                       tokenize=False,
                                       return_tensors=None)
    # ↓↓↓ 手写模板 ↓↓↓
    sys_tk   = "<reserved_106>你是洁净室工程知识助手，请基于已知信息准确回答。<reserved_107>"
    user_tk  = "<reserved_102>"
    user_end = "<reserved_103>"
    bot_tk   = "<reserved_104>"
    # bot_end = "<reserved_105>"   # 不需要提前加，模型生成时会补
    return (
        f"{sys_tk}"
        f"{user_tk}已知信息：\n{context}\n\n问题：{question}{user_end}"
        f"{bot_tk}"
    )

#  5) 交互式问答
while True:
    q = input("\n问题> ").strip()
    if not q:
        break

    # 相似度检索
    q_vec = emb_model.encode([q])
    _, I = index.search(q_vec, k=4)
    context = "\n".join(chunks[i] for i in I[0])

    prompt = build_prompt(q, context)                  # ← 用模板包装
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,                                      # ← 传入映射而非张量
        max_new_tokens=256,
        do_sample=True)                               # 可改 True+温度采样
    ans = tok.decode(out[0][inputs.input_ids.shape[-1]:],
                     skip_special_tokens=True).strip()

    print("\n回答:", ans) #回答


