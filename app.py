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
- Answer based on the provided law sections
- Always mention which law/section you are referencing
- Never say someone is guilty or innocent
- Keep answers simple and human
- If information comes from Indian Kanoon, mention it naturally
- If you truly don't have enough information, say: "I don't have complete information on this. Please visit indiankanoon.org or consult a lawyer."
- NEVER make up or guess legal information
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

async def search_indian_kanoon(query: str) -> str:
    """Search Indian Kanoon API as fallback"""
    if not IK_TOKEN:
        return ""
    try:
        import urllib.parse
        import re
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.indiankanoon.org/search/?formInput={encoded_query}&pagenum=0"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Token {IK_TOKEN}"}
            )

        if response.status_code != 200:
            print(f"Indian Kanoon error: {response.status_code}")
            return ""

        data = response.json()
        docs = data.get("docs", [])

        if not docs:
            return ""

        context = "--- Information from Indian Kanoon ---\n"
        for doc in docs[:3]:
            title = doc.get("title", "")
            headline = doc.get("headline", "")
            docsource = doc.get("docsource", "")
            headline_clean = re.sub(r'<[^>]+>', '', headline)
            context += f"\n[{docsource} — {title}]\n{headline_clean}\n"

        return context

    except Exception as e:
        print(f"Indian Kanoon search error: {e}")
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
        # Step 1 — Search local FAISS database
        results_with_dist = search_law(q.question)
        low_confidence = is_low_confidence(results_with_dist)

        # Build local context
        local_context = ""
        for dist, law in results_with_dist:
            local_context += f"\n[{law['source']} - {law['section']}]\n{law['content']}\n"

        # Step 2 — If low confidence OR constitution query, call Indian Kanoon
        ik_context = ""
        if needs_indian_kanoon(q.question, low_confidence):
            print(f"🔍 Calling Indian Kanoon for: {q.question}")
            ik_context = await search_indian_kanoon(q.question)
            if ik_context:
                print("✅ Indian Kanoon returned results!")

        # Step 3 — Combine contexts
        context = local_context
        if ik_context:
            context += f"\n{ik_context}"

        if not context.strip():
            context = "No relevant law sections found in database."

        # Step 4 — Detect language
        hindi_chars = sum(1 for c in q.question if '\u0900' <= c <= '\u097F')
        language = "Hindi" if hindi_chars > 2 else "English"

        # Step 5 — Call Groq LLM
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"IMPORTANT: Reply strictly in {language} only.\n\nLaw sections:\n{context}\n\nQuestion: {q.question}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        answer = response.choices[0].message.content


        # Step 6 — Save to Supabase
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
