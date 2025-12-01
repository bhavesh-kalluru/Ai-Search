# 🧠 AI/ML Job Finder (Streamlit + Perplexity)

> I’m actively looking for a **full-time role in Generative AI / AI / Machine Learning** (US-based, remote or hybrid).  
> If this project aligns with what you’re building, I’d love to connect and contribute!

---

A Streamlit app that searches the **live web** (via the Perplexity API) for roles in:

- Generative AI / LLMs
- Machine Learning / Deep Learning
- Data Engineering
- Data Analysis / Analytics

focused on candidates with **2–5 years of experience**.

You can use it daily to discover:

- 🔹 Full-time, Part-time, Contract, Internship, Temporary roles  
- 🔹 Remote, Onsite, and Hybrid jobs  
- 🔹 Openings from job boards (LinkedIn, Naukri, Dice, ZipRecruiter, Indeed, Glassdoor, etc.)  
- 🔹 Roles directly from company career pages (Greenhouse, Lever, Ashby, etc.)

---

## ✨ Features

- 🔍 **Live job search** using Perplexity’s browsing capabilities  
- 🎯 Filters by:
  - Experience: **2, 3, 4, 5 years**
  - Domains: **GenAI, ML, DL, Data Engineering, Data Analytics**
  - Work mode: **Remote, Onsite, Hybrid**
  - Employment type: **Full-time, Part-time, Contract, Internship, Temporary**
- 🏢 Shows **Company**, **Role**, **Location**, **Posted date**, **Source site**, and **direct job link**
- 📊 Results in an interactive **Streamlit table**
- ⬇️ **Download CSV** of filtered jobs
- 🧠 Prioritizes **remote** roles at the top of the list
- 🔧 Optional **OpenAI API** integration to clean / repair JSON if needed

---

## 🧑‍💻 Why I Built This

I’m building a career in **GenAI / AI / ML** and wanted a practical project that:

- Uses **real APIs** (Perplexity + optional OpenAI)
- Works with **real-world data** (live job postings from the web)
- Demonstrates skills in:
  - Python
  - Streamlit
  - Prompt engineering
  - Data wrangling with Pandas
  - API integration and error handling

This repository is both a **tool I use daily** and a **portfolio project** to showcase my abilities for a **full-time role in GenAI / AI / ML / Data**.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
2. (Optional) Create a virtual environment
bash
Copy code
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# On Windows: .venv\Scripts\activate
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
🔑 API Keys (No .env Required)
This project does not require a .env file.

You have two options:

Option A – Type keys directly in the app (simplest)
When you run the app (see below), the sidebar will show:
Perplexity API Key (required)
OpenAI API Key (optional)

You can manually paste your keys into those fields each time you start the app.
Nothing is written to disk, so nothing secret gets committed to GitHub.

Option B – Streamlit secrets (optional, for convenience)
If you prefer, you can use secrets.toml (this file should NOT be committed):

Create:

bash
Copy code
mkdir -p .streamlit
nano .streamlit/secrets.toml
With:

toml
Copy code
PERPLEXITY_API_KEY = "your_perplexity_key"
OPENAI_API_KEY = "your_openai_key"  # optional
Streamlit will read these automatically and prefill the sidebar fields.

🔐 Important: Never commit real API keys. Keep .streamlit/secrets.toml out of Git.

🏃 Run the App
From the project directory:

bash
Copy code
streamlit run main.py
Then open the URL shown in the terminal (usually: http://localhost:8501).
Paste your Perplexity API key into the sidebar (or let secrets prefill it).
(Optional) Paste your OpenAI API key if you want auto-fixing of malformed JSON.

Choose:
Experience years (2–5)
Domains (GenAI, ML, DL, Data Eng, Analytics)
Work mode (Remote / Onsite / Hybrid)
Employment type (Full-time / Part-time / Contract / Internship / Temporary)
Extra keywords (e.g., RAG, LLMOps, PyTorch, TensorFlow, Databricks, Snowflake)
Click “🔎 Search now”.

You’ll get a table of jobs with links you can open in a new tab, plus a Download CSV button.

🧩 How It Works (High Level)
The app builds a prompt describing:
Experience range
Domains (GenAI / ML / DL / Data)
Work modes & employment types
Preferred job sources (LinkedIn, Naukri, Dice, ZipRecruiter, etc.)
Look-back window (e.g., last 14 days)
It calls the Perplexity /chat/completions API (e.g. with model="sonar-pro") and asks for a pure JSON response:

jsonc
Copy code
{
  "results": [
    {
      "company": "Example Corp",
      "role": "Machine Learning Engineer",
      "experience_required": "3+ years",
      "domain": "Machine Learning",
      "work_mode": "Remote",
      "employment_type": "Full-time",
      "location": "Remote - US",
      "posted_date": "2025-11-29",
      "source": "LinkedIn",
      "url": "https://..."
    }
  ]
}
The app:

Normalizes different possible company fields (company, company_name, employer, etc.)
De-duplicates entries (same company + role + URL)
Prioritizes Remote roles
Applies secondary filters (experience, domain, work mode, type, date)
Displays the result in a Streamlit dataframe with a clickable Open link
If the JSON is slightly messy, and the OpenAI key is provided, the app can:
Send the raw text to OpenAI
Ask it to repair the JSON
Parse and display the fixed data

🛠 Tech Stack
Language: Python
Frontend / UI: Streamlit
APIs:
Perplexity API for live web search
OpenAI API (optional) for JSON fixing
Data: Pandas for tabular processing
Environment: macOS-friendly, works cross-platform

🙌 Open to Opportunities
I’m actively seeking a full-time role in:
Generative AI / LLMs
Applied Machine Learning
AI Engineering / MLOps
Data Science / ML Engineering

If you’re hiring or working on something interesting in this space, feel free to:
Open an issue or discussion on this repo
Reach out via LinkedIn(https://www.linkedin.com/in/bhaveshkalluru/) / email (kallurubhavesh341@gmail.com)
I’d love to collaborate on real-world GenAI products and systems.

🏷 Hashtags
#GenAI #GenerativeAI #AI #MachineLearning #ML #DeepLearning
#LLM #LLMs #MLOps #DataEngineering #DataScience #DataAnalytics
#Python #Streamlit #PerplexityAI #OpenAI
#AIJobs #TechJobs #RemoteJobs #JobSearch #JobHunting
#FullTimeRole #Hiring #CareerInAI #AIPortfolio #AIProjects