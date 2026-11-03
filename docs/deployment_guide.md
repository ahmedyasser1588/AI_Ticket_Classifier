# 🚀 Deployment Guide — AI Ticket Classifier

> رفع المشروع على HuggingFace Spaces مجاناً خطوة بخطوة.

---

## الملفات اللي اتضافت (بدون تغيير في كودك الأصلي)

| الملف | الغرض |
|---|---|
| [`app.py`](file:///d:/Learning/Cources/Depi_Agentic/Ticket_Classifier_Depi-main/New%20folder/ME/app.py) | 🎨 Gradio UI — entry point للـ HuggingFace |
| [`requirements_deployment.txt`](file:///d:/Learning/Cources/Depi_Agentic/Ticket_Classifier_Depi-main/New%20folder/ME/requirements_deployment.txt) | 📦 Dependencies للـ deployment |
| [`README.md`](file:///d:/Learning/Cources/Depi_Agentic/Ticket_Classifier_Depi-main/New%20folder/ME/README.md) | 📄 صفحة المشروع (بـ YAML frontmatter لـ HF Spaces) |
| [`.gitignore`](file:///d:/Learning/Cources/Depi_Agentic/Ticket_Classifier_Depi-main/New%20folder/ME/.gitignore) | 🔒 يحمي الـ `.env` من الـ git |
| [`.env.example`](file:///d:/Learning/Cources/Depi_Agentic/Ticket_Classifier_Depi-main/New%20folder/ME/.env.example) | 📋 Template للـ env variables |

---

## خطوات الـ Deployment

### الخطوة 0 — إنشاء Groq API Key (مجاني)

1. اذهب إلى [console.groq.com](https://console.groq.com)
2. سجّل بـ Google أو GitHub
3. من القائمة: **API Keys → Create API Key**
4. احفظ الـ key (هتحتاجه في الخطوة 3)

---

### الخطوة 1 — إنشاء GitHub Repository

```bash
# في مجلد ME:
git init
git add app.py README.md requirements_deployment.txt .gitignore .env.example
git add prompts/ structured/ LLM/ evaluation/ src/ Data/
git commit -m "feat: AI Ticket Classifier with Gradio deployment"

# ارفع على GitHub
# أنشئ repo جديد على github.com ثم:
git remote add origin https://github.com/YOUR_USERNAME/ticket-classifier.git
git push -u origin main
```

> [!CAUTION]
> تأكد إن `.env` **مش موجود** في الـ `git add` — الـ `.gitignore` هيحميه تلقائياً

---

### الخطوة 2 — إنشاء HuggingFace Space

1. اذهب إلى [huggingface.co/new-space](https://huggingface.co/new-space)
2. اختر الإعدادات دي:

```
Space Name:  ticket-classifier
License:     MIT
SDK:         Gradio          ← مهم
Visibility:  Public          ← للبورتفوليو
```

3. اضغط **Create Space**

---

### الخطوة 3 — إضافة الـ API Key كـ Secret

في صفحة الـ Space:

```
Settings → Variables and Secrets → New Secret
```

أضف 3 secrets:

| Name | Value |
|---|---|
| `LLM_api_key` | `gsk_xxxxxxxxxxxx` (من Groq) |
| `LLM_base_url` | `https://api.groq.com/openai/v1` |
| `LLM_Model` | `llama-3.1-8b-instant` |

---

### الخطوة 4 — رفع الكود

```bash
# أضف HuggingFace كـ remote ثاني
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/ticket-classifier

# ارفع على HuggingFace
git push hf main
```

أو من الـ GitHub مباشرة عن طريق **Files → Upload files** في صفحة الـ Space.

> [!NOTE]
> HuggingFace هيقرأ الـ `README.md` ويأخذ منه الـ YAML frontmatter تلقائياً ويعمل setup للـ Space.

---

### الخطوة 5 — انتظر Build (2-3 دقايق)

HuggingFace هيـ:
1. يقرأ `requirements_deployment.txt` ويثبّت الـ packages
2. يشغّل `app.py`
3. يعرض الـ Gradio UI على رابط:

```
https://YOUR_USERNAME-ticket-classifier.hf.space
```

---

## اختبار محلي قبل الرفع

```bash
# في مجلد ME:
venv\Scripts\activate          # Windows

pip install gradio openai python-dotenv

python app.py
# افتح: http://127.0.0.1:7860
```

---

## صورة الـ App النهائية

```
┌─────────────────────────────────────────────────┐
│           🎫 AI Ticket Classifier               │
│   Powered by Groq & Llama 3.1                  │
├─────────────────────────────────────────────────┤
│ 📩 Support Ticket                               │
│ ┌─────────────────────────────────────────────┐ │
│ │ I was charged twice for my subscription...  │ │
│ └─────────────────────────────────────────────┘ │
│           [ ✨ Classify Ticket ]                │
├──────────────┬──────────────┬───────────────────┤
│ 🏷️ Category  │ 💬 Sentiment │ 🚨 Urgency        │
│ 💳 Billing   │ 😠 Negative  │ 🔴 High           │
├─────────────────────────────────────────────────┤
│ 📝 AI Summary                                  │
│ Customer was double-charged and wants refund.  │
├─────────────────────────────────────────────────┤
│ 💡 Try an example:                             │
│ [My laptop screen...] [I was charged twice...] │
│ [I forgot my password...] [Package late...]    │
└─────────────────────────────────────────────────┘
```

---

## رابط البورتفوليو النهائي

```
https://huggingface.co/spaces/YOUR_USERNAME/ticket-classifier
```

ضعه في:
- ✅ GitHub Profile README
- ✅ LinkedIn (Projects Section)
- ✅ CV / Resume
- ✅ Portfolio Website

---

## ملاحظات مهمة

> [!WARNING]
> الـ Groq API Key الموجودة في الـ `.env` **موجودة في ملف git tracked** — يفضل تعمل key جديدة بعد ما ترفع المشروع، أو تعمل revoke للقديمة من console.groq.com

> [!TIP]
> لو الـ Space "went to sleep" بعد فترة عدم استخدام، أول request هيصحّيه تلقائياً في 10-30 ثانية — دا طبيعي في الـ free tier.
