from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from collections import defaultdict
import time
import os
import httpx

load_dotenv()

# ── DOWNLOAD DATA FROM HUGGING FACE IF NOT EXISTS ──
def download_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/law_index.faiss"):
        print("📥 Downloading data from Hugging Face...")
        from huggingface_hub import hf_hub_download
        files = ["law_index.faiss", "chunks_meta.pkl", "all_chunks.json"]
        for f in files:
            print(f"  Downloading {f}...")
            hf_hub_download(
                repo_id="anuragsingh111/Lex-AI",
                repo_type="dataset",
                filename=f,
                local_dir="data"
            )
        print("✅ Data downloaded!")
    else:
        print("✅ Data already exists!")

download_data()

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

app = FastAPI()

print("🚀 Loading models...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
index = faiss.read_index("data/law_index.faiss")
with open("data/chunks_meta.pkl", 'rb') as f:
    chunks = pickle.load(f)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

IK_TOKEN = os.getenv("IK_TOKEN")

print("✅ Lex AI is ready!")

rate_store = defaultdict(list)

def is_rate_limited(user_id: str) -> bool:
    now = time.time()
    rate_store[user_id] = [t for t in rate_store[user_id] if now - t < 3600]
    if len(rate_store[user_id]) >= 10:
        return True
    rate_store[user_id].append(now)
    return False

SYSTEM_PROMPT = """You are Lex AI, an Indian legal information assistant.
You help Indian citizens understand their legal rights in simple, clear language.

LANGUAGE RULE:
- If user writes in English reply in English
- If user writes in Hindi reply in Hindi
- NEVER mention language in your response

ANSWER RULES:
- ONLY answer using the provided law sections and context below
- NEVER use your own training knowledge — ONLY the provided context
- Always mention which law/section you are referencing
- Never say someone is guilty or innocent
- Keep answers simple and human
- If the provided context does not contain enough information say EXACTLY:
  "I don't have reliable information on this specific topic. Please visit indiankanoon.org for accurate legal information or call NALSA free legal aid: 15100"
- NEVER guess, assume, or make up any legal information
- NEVER reference laws or sections not present in the provided context
- Do NOT add any disclaimer at the end"""

class Question(BaseModel):
    question: str

class UserSession(BaseModel):
    user_id: str
    email: str
    name: str

def search_law(query, top_k=7):
    expansions = {
        'fir': 'first information report police complaint cognizable offence',
        'arrest': 'arrest rights detained custody police',
        'bail': 'bail bond release custody bailable',
        'murder': 'murder punishment homicide death',
        'consumer': 'consumer protection defective product complaint',
        'divorce': 'divorce marriage dissolution hindu marriage act',
        'rti': 'right to information public authority application',
        'accident': 'motor vehicle accident compensation insurance',
        'drugs': 'ndps narcotic drugs psychotropic substances',
        'wages': 'minimum wages labour payment salary',
        'cybercrime': 'cyber crime IT act online fraud hacking',
        'domestic violence': 'domestic violence women protection act',
        'article': 'constitution of india fundamental rights article',
        'constitution': 'constitution of india fundamental rights articles',
        '370': 'article 370 jammu kashmir constitution',
        '21': 'article 21 right to life personal liberty constitution',
        '19': 'article 19 freedom of speech expression constitution',
    }
    expanded = query
    for keyword, expansion in expansions.items():
        if keyword in query.lower():
            expanded = query + " " + expansion
            break

    embedding = embedder.encode([expanded])
    distances, indices = index.search(np.array(embedding, dtype=np.float32), top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((dist, chunks[idx]))
    return results

def is_low_confidence(results, threshold=1.0):
    """Check if FAISS results are low confidence"""
    if not results:
        return True
    best_distance = results[0][0]
    return best_distance > threshold

def needs_indian_kanoon(question: str, low_confidence: bool) -> bool:
    """Check if we should call Indian Kanoon"""
    # Always call for constitution articles
    constitution_triggers = [
        'article ', 'article-', 'constitution',
        'fundamental right', 'directive principle',
        'amendment', 'schedule', 'preamble'
    ]
    q_lower = question.lower()
    for trigger in constitution_triggers:
        if trigger in q_lower:
            return True
    # Also call if FAISS is not confident
    return low_confidence

async def fetch_ik_document(tid: int) -> str:
    """Fetch full document from Indian Kanoon by doc ID"""
    try:
        import re
        url = f"https://api.indiankanoon.org/doc/{tid}/"
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Token {IK_TOKEN}"}
            )
        if response.status_code != 200:
            return ""
        data = response.json()
        doc_html = data.get("doc", "")
        doc_text = re.sub(r'<[^>]+>', ' ', doc_html)
        doc_text = re.sub(r'\s+', ' ', doc_text).strip()
        return doc_text[:2000]
    except Exception as e:
        print(f"IK doc fetch error: {e}")
        return ""

async def search_indian_kanoon(query: str, constitution_mode: bool = False) -> str:
    """Search Indian Kanoon API"""
    if not IK_TOKEN:
        return ""
    try:
        import urllib.parse
        import re

        if constitution_mode:
            search_query = query + " doctypes:laws"
        else:
            search_query = query

        encoded_query = urllib.parse.quote(search_query)
        url = f"https://api.indiankanoon.org/search/?formInput={encoded_query}&pagenum=0"

        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Token {IK_TOKEN}"}
            )

        if response.status_code != 200:
            return ""

        data = response.json()
        docs = data.get("docs", [])
        if not docs:
            return ""

        context = "--- Indian Kanoon ---\n"

        if constitution_mode:
            # Find the actual Constitution document first
            constitution_doc = None
            for doc in docs:
                docsource = doc.get("docsource", "")
                title = doc.get("title", "")
                if "Constitution" in docsource or "Constitution" in title:
                    constitution_doc = doc
                    break

            if constitution_doc:
                tid = constitution_doc.get("tid")
                title = constitution_doc.get("title", "")
                if tid:
                    full_text = await fetch_ik_document(tid)
                    if full_text:
                        context += f"\n[Constitution of India — {title}]\n{full_text}\n"
                        return context

            # Fallback to headlines
            for doc in docs[:2]:
                title = doc.get("title", "")
                headline = doc.get("headline", "")
                docsource = doc.get("docsource", "")
                headline_clean = re.sub(r'<[^>]+>', '', headline).strip()
                context += f"\n[{docsource} — {title}]\n{headline_clean}\n"
        else:
            for doc in docs[:2]:
                title = doc.get("title", "")
                headline = doc.get("headline", "")
                docsource = doc.get("docsource", "")
                headline_clean = re.sub(r'<[^>]+>', '', headline).strip()
                context += f"\n[{docsource} — {title}]\n{headline_clean}\n"

        return context

    except Exception as e:
        print(f"Indian Kanoon error: {e}")
        return ""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(open("templates/index.html", encoding="utf-8").read())

@app.post("/auth/session")
async def save_session(user: UserSession):
    try:
        supabase.table("users").upsert({
            "id": user.user_id,
            "email": user.email,
            "name": user.name,
        }).execute()
        response = JSONResponse({"status": "ok"})
        response.set_cookie("user_id", user.user_id, httponly=True, max_age=604800, samesite="lax")
        response.set_cookie("user_email", user.email, max_age=604800, samesite="lax")
        response.set_cookie("user_name", user.name, max_age=604800, samesite="lax")
        return response
    except Exception as e:
        print(f"Session error: {e}")
        return JSONResponse({"status": "error"})

@app.get("/auth/logout")
async def logout():
    response = RedirectResponse("/")
    response.delete_cookie("user_id")
    response.delete_cookie("user_email")
    response.delete_cookie("user_name")
    return response

@app.get("/auth/user")
async def get_user(request: Request):
    user_id = request.cookies.get("user_id")
    if user_id:
        return JSONResponse({
            "logged_in": True,
            "user_id": user_id,
            "email": request.cookies.get("user_email"),
            "name": request.cookies.get("user_name")
        })
    return JSONResponse({"logged_in": False})

@app.post("/ask")
async def ask(q: Question, request: Request):
    user_id = request.cookies.get("user_id") or request.client.host

    if is_rate_limited(user_id):
        return JSONResponse({
            "answer": "⚠️ You have reached the limit of 10 questions per hour. Please try again later.\n\nFree legal aid: NALSA Helpline 15100",
            "conversation_id": None,
            "status": "rate_limited"
        })

    sensitive = ['pocso', 'child abuse', 'minor', 'sexual assault', 'bachche', 'bacha']
    if any(w in q.question.lower() for w in sensitive):
        answer = """This case involves the POCSO Act (Protection of Children from Sexual Offences Act, 2012).

⚠️ This is a highly sensitive and serious legal matter. An AI system cannot guide you through this — you MUST consult a real lawyer immediately.

📌 What the law says:
- POCSO Act 2012 covers sexual offences against children under 18
- Both accused and victims have specific legal rights
- Bail conditions are very strict under POCSO
- Cases are tried in Special Courts only

🆘 Get help RIGHT NOW:
- NALSA Free Legal Aid: 15100
- Police helpline: 100
- Child helpline: 1098

Please do not delay — contact a lawyer today."""
        conv_id = None
        real_user_id = request.cookies.get("user_id")
        if real_user_id:
            try:
                result = supabase.table("conversations").insert({
                    "user_id": real_user_id,
                    "question": q.question,
                    "answer": answer
                }).execute()
                conv_id = result.data[0]['id'] if result.data else None
            except:
                pass
        return JSONResponse({"answer": answer, "conversation_id": conv_id, "status": "ok"})

    try:
        import re as re2
        # Detect if this is a Constitution article query
        article_match = re2.search(r'article\s+(\d+)', q.question.lower())
        is_constitution_query = article_match is not None

        # Step 1 — Skip FAISS for constitution articles, use directly for others
        local_context = ""
        if not is_constitution_query:
            results_with_dist = search_law(q.question)
            low_confidence = is_low_confidence(results_with_dist)
            for dist, law in results_with_dist:
                local_context += f"\n[{law['source']} - {law['section']}]\n{law['content']}\n"
        else:
            low_confidence = True

        # Step 2 — Call Indian Kanoon when needed
        ik_context = ""
        if needs_indian_kanoon(q.question, low_confidence):
            print(f"Calling Indian Kanoon for: {q.question}")
            ik_context = await search_indian_kanoon(q.question, constitution_mode=is_constitution_query)

        # Step 3 — Build context smartly
        if is_constitution_query and ik_context:
            context = ik_context  # ONLY use Indian Kanoon for articles
        else:
            context = local_context
            if ik_context:
                context += f"\n{ik_context}"

        # Context validation — don't call LLM if context is too poor
        if not context.strip() or len(context.strip()) < 100:
            return JSONResponse({
                "answer": "I don't have reliable information on this specific topic. Please visit indiankanoon.org for accurate legal information or call NALSA free legal aid: 15100",
                "conversation_id": None,
                "status": "ok"
            })

        # Step 4 — Detect language
        hindi_chars = sum(1 for c in q.question if '\u0900' <= c <= '\u097F')
        language = "Hindi" if hindi_chars > 2 else "English"

        # Step 5 — Build prompt (special for constitution articles)
        if is_constitution_query and article_match:
            article_num = article_match.group(1)
            user_prompt = f"IMPORTANT: Reply strictly in {language} only.\n\nThe user is asking about Article {article_num} of the Constitution of India. Use ONLY the information below. Do NOT reference RTI or any other unrelated law.\n\nInformation:\n{context}\n\nQuestion: {q.question}"
        else:
            user_prompt = f"IMPORTANT: Reply strictly in {language} only.\n\nLaw sections:\n{context}\n\nQuestion: {q.question}"

        # Step 6 — Call Groq LLM
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )

        answer = response.choices[0].message.content

        # Step 7 — Save to Supabase
        conv_id = None
        real_user_id = request.cookies.get("user_id")
        if real_user_id:
            try:
                result = supabase.table("conversations").insert({
                    "user_id": real_user_id,
                    "question": q.question,
                    "answer": answer
                }).execute()
                conv_id = result.data[0]['id'] if result.data else None
            except:
                pass

        return JSONResponse({"answer": answer, "conversation_id": conv_id, "status": "ok"})

    except Exception as e:
        print(f"Ask error: {e}")
        return JSONResponse({"answer": "Something went wrong. Please try again.", "conversation_id": None, "status": "error"})

@app.post("/feedback")
async def feedback(request: Request):
    try:
        body = await request.json()
        user_id = request.cookies.get("user_id")
        if not user_id:
            return JSONResponse({"status": "not_logged_in"})
        supabase.table("feedback").insert({
            "user_id": user_id,
            "conversation_id": body.get("conversation_id"),
            "rating": body.get("rating")
        }).execute()
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"status": "error"})

# ── CASE ANALYSIS FEATURE ──

class CaseDetails(BaseModel):
    what_happened: str
    role: str          # "victim" or "accused"
    fir_filed: str     # "yes", "no", "not sure"
    state: str
    charges: str       # optional, can be empty

ANALYZE_PROMPT = """You are Lex AI, an expert Indian legal analyst.

A person has described their legal situation. Analyze it thoroughly and provide:

1. 📋 CASE SUMMARY — Summarize the situation in 2-3 lines
2. ⚖️ APPLICABLE LAWS — Which Indian laws and sections apply
3. 💪 STRONGEST ARGUMENTS — Top 3 legal arguments for their position
4. ⚠️ RISKS & CHALLENGES — What they should be careful about
5. 📝 RECOMMENDED STEPS — Exact step-by-step actions to take
6. 🆘 WHEN TO GET A LAWYER — Be specific about when professional help is essential

RULES:
- Base analysis on provided law sections and Indian Kanoon data
- Be specific — mention actual section numbers
- Never declare guilt or innocence
- Be practical and actionable
- Match the language of the user (Hindi/English)
- Do NOT add disclaimer at the end"""

@app.post("/analyze")
async def analyze_case(case: CaseDetails, request: Request):
    user_id = request.cookies.get("user_id") or request.client.host

    if is_rate_limited(user_id):
        return JSONResponse({
            "analysis": "⚠️ Rate limit reached. Please try again after an hour.\n\nFree legal aid: NALSA Helpline 15100",
            "status": "rate_limited"
        })

    try:
        # Build full case description
        full_case = f"""
Situation: {case.what_happened}
Role: {case.role}
FIR Filed: {case.fir_filed}
State: {case.state}
Charges mentioned: {case.charges if case.charges else 'Not specified'}
"""

        # Search FAISS for relevant laws
        results = search_law(case.what_happened, top_k=10)
        local_context = ""
        for dist, law in results:
            local_context += f"\n[{law['source']} - {law['section']}]\n{law['content']}\n"

        # Also search Indian Kanoon for relevant cases
        ik_context = await search_indian_kanoon(case.what_happened)

        # Combine context
        context = local_context
        if ik_context:
            context += f"\n{ik_context}"

        # Detect language
        hindi_chars = sum(1 for c in case.what_happened if '\u0900' <= c <= '\u097F')
        language = "Hindi" if hindi_chars > 2 else "English"

        # Call Groq for analysis
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": ANALYZE_PROMPT},
                {"role": "user", "content": f"Reply in {language} only.\n\nCase Details:\n{full_case}\n\nRelevant Law Sections:\n{context}\n\nProvide complete legal analysis."}
            ],
            temperature=0.0,
            max_tokens=2000
        )

        analysis = response.choices[0].message.content

        # Save to Supabase
        conv_id = None
        real_user_id = request.cookies.get("user_id")
        if real_user_id:
            try:
                result = supabase.table("conversations").insert({
                    "user_id": real_user_id,
                    "question": f"[CASE ANALYSIS] {case.what_happened[:200]}",
                    "answer": analysis
                }).execute()
                conv_id = result.data[0]['id'] if result.data else None
            except:
                pass

        return JSONResponse({
            "analysis": analysis,
            "conversation_id": conv_id,
            "status": "ok"
        })

    except Exception as e:
        print(f"Analyze error: {e}")
        return JSONResponse({
            "analysis": "Something went wrong. Please try again.",
            "status": "error"
        })
