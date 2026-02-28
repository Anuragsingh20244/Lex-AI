from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv
from collections import defaultdict
import time
import os

load_dotenv()

app = FastAPI()

print("🚀 Loading models...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
index = faiss.read_index("data/vectordb/law_index.faiss")
with open("data/vectordb/chunks_meta.pkl", 'rb') as f:
    chunks = pickle.load(f)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

print("✅ Lex AI is ready!")

# Rate limiting — 10 requests per hour per user
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
- Just directly answer naturally

ANSWER RULES:
- Only answer based on the provided law sections
- Always mention which law/section you are referencing
- Never say someone is guilty or innocent
- Keep answers simple and human
- If someone says they are falsely accused, acknowledge their situation with empathy first
- Do NOT add any disclaimer at the end"""

class Question(BaseModel):
    question: str

def search_law(query, top_k=7):
    expansions = {
        'fir': 'first information report police complaint cognizable offence',
        'arrest': 'arrest rights detained custody police',
        'bail': 'bail bond release custody bailable',
        'murder': 'murder punishment homicide death',
        'complaint': 'complaint magistrate police report filing',
        'rights': 'fundamental rights constitution protection',
        'cybercrime': 'cyber crime IT act online fraud hacking',
        'domestic violence': 'domestic violence women protection act',
        'consumer': 'consumer protection defective product complaint',
        'divorce': 'divorce marriage dissolution hindu marriage act',
        'rti': 'right to information public authority application',
        'accident': 'motor vehicle accident compensation insurance',
        'drugs': 'ndps narcotic drugs psychotropic substances',
        'wages': 'minimum wages labour payment salary',
    }
    expanded = query
    for keyword, expansion in expansions.items():
        if keyword in query.lower():
            expanded = query + " " + expansion
            break
    embedding = embedder.encode([expanded])
    distances, indices = index.search(np.array(embedding, dtype=np.float32), top_k)
    return [chunks[idx] for idx in indices[0]]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTMLResponse(open("templates/index.html", encoding="utf-8").read())

@app.get("/auth/login")
async def login():
    try:
        result = supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": "http://localhost:8000/auth/callback",
                "scopes": "email profile"
            }
        })
        return RedirectResponse(result.url)
    except Exception as e:
        print(f"Login error: {e}")
        return RedirectResponse("/")

@app.get("/auth/callback")
async def auth_callback(request: Request):
    code = request.query_params.get("code")
    if code:
        try:
            result = supabase.auth.exchange_code_for_session({"auth_code": code})
            user = result.user
            supabase.table("users").upsert({
                "id": user.id,
                "email": user.email,
                "name": user.user_metadata.get("full_name", ""),
                "avatar_url": user.user_metadata.get("avatar_url", "")
            }).execute()
            response = RedirectResponse("/?logged_in=true")
            response.set_cookie("user_id", user.id, httponly=True, max_age=604800)
            response.set_cookie("user_email", user.email, max_age=604800)
            response.set_cookie("user_name", user.user_metadata.get("full_name", "User"), max_age=604800)
            return response
        except Exception as e:
            print(f"Auth error: {e}")
            return RedirectResponse("/")
    return RedirectResponse("/")

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
    user_email = request.cookies.get("user_email")
    user_name = request.cookies.get("user_name")
    if user_id:
        return JSONResponse({
            "logged_in": True,
            "user_id": user_id,
            "email": user_email,
            "name": user_name
        })
    return JSONResponse({"logged_in": False})

@app.post("/ask")
async def ask(q: Question, request: Request):
    user_id = request.cookies.get("user_id") or request.client.host

    # Rate limiting
    if is_rate_limited(user_id):
        return JSONResponse({
            "answer": "⚠️ You have reached the limit of 10 questions per hour. Please try again later.\n\nFree legal aid: NALSA Helpline 15100",
            "conversation_id": None,
            "status": "rate_limited"
        })

    # POCSO special handling
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
- Nearest District Legal Services Authority (DLSA)

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

    # Normal flow
    try:
        relevant_laws = search_law(q.question)
        context = ""
        for law in relevant_laws:
            context += f"\n[{law['source']} - {law['section']}]\n{law['content']}\n"

        hindi_chars = sum(1 for c in q.question if '\u0900' <= c <= '\u097F')
        language = "Hindi" if hindi_chars > 2 else "English"

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

        # Save conversation to DB if logged in
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