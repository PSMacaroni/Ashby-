# Colon Tumor Front Analyzer (Ashby Prize Hackathon 2026)

A pathologist uploads a colon cancer whole-slide image (an H&E-stained microscope slide) and our app shows two things. First, a **color-coded tissue map** (tumor, normal lining, muscle, fat, immune cells, mucus, dead tissue, stroma = the connective tissue around the tumor) that shows the model's probabilities when you hover. Second, a **summary panel**: tumor %, how deep the tumor has grown through the colon wall, whether the border is smooth or ragged, immune cells at the tumor center vs. edge, any dead core, and the high-risk features that matter when deciding on chemo for stage II colon cancer. Under the hood we use three pathology AI models (CONCH, MUSK, PathGen-CLIP) "zero-shot", meaning we describe each tissue type in words and the model matches image patches to the closest description. We add small classifiers on top. The heavy GPU work runs on NCSA's Delta supercomputer and saves small result files, a Streamlit app displays them, and NCSA Lumen can optionally write a plain-language summary.

## Tools in plain English

| Tool | What it is | What we use it for |
|---|---|---|
| **GitHub** | A website that stores our code and its history | Sharing code so all 4 of us work on the same project |
| **Hugging Face** | The "GitHub for AI models" | Downloading CONCH, MUSK, PathGen-CLIP (you have to request access to each) |
| **Delta** | NCSA's supercomputer, with lots of powerful GPUs | Running the models on big slides (way too slow on a laptop) |
| **ACCESS** | The national program that hands out supercomputer time | Your ACCESS account is how you log into Delta |
| **Open OnDemand** | A website for using Delta from your browser | Starting a Jupyter session on Delta without the command line |
| **Jupyter** | Notebooks where you run code one cell at a time and see results right away | Trying out models, looking at tiles |
| **Lumen** | NCSA's hosted AI chatbot service (works like the OpenAI API) | Optional: turning our numbers into a readable summary |
| **Streamlit** | A Python library that turns a script into a web app | Our demo app |
| **Slack** | Group chat | Talking to teammates and organizers. **Never paste tokens here.** |

## Tonight (before Saturday)

- [ ] **Hugging Face**: make an account, then on each page click "Agree and access" / request access. Wait until it says **approved**:
  - https://huggingface.co/MahmoodLab/CONCH
  - https://huggingface.co/xiangjx/musk
  - https://huggingface.co/jamessyx/PathGen-CLIP
  - https://huggingface.co/datasets/jamessyx/PathGen
- [ ] **Hugging Face token**: Settings → Access Tokens → New token → type **Read**. Save it somewhere private.
- [ ] **ACCESS / Delta**: make sure you have an ACCESS account and you've been added to the hackathon's Delta allocation. Set up **NCSA Duo** (the two-factor login app). Delta won't let you in without it.
- [ ] **Lumen**: get your API key (and the base URL) from the organizers.
- [ ] **Your computer**: install [GitHub Desktop](https://desktop.github.com/) and [VS Code](https://code.visualstudio.com/), then clone this repo.

## On your laptop (optional, for the app and small tests)

A **virtual environment** (`.venv`) is a private box of Python packages just for this project, so it can't clash with anything else on your computer.

```bash
python -m venv .venv                     # make the box (once)
.venv\Scripts\activate                   # step into it - Windows
source .venv/bin/activate                # step into it - Mac
pip install -r requirements.txt          # fill it (once; torch is big, takes a while)
cp .env.example .env                     # then paste your tokens into .env
python scripts/check_setup.py
```

Activate it every time you open a new terminal (you'll see `(.venv)` at the start of the line). "No GPU" is expected on a laptop.

## Saturday on Delta

1. Log into **Open OnDemand** for Delta (use your ACCESS login + Duo).
2. Start an **Interactive App → Jupyter** session. Pick a **GPU** partition (e.g. `gpuA100x4`), 1 GPU, a few hours.
3. In Jupyter, open a **Terminal** and clone the repo. GitHub doesn't accept your normal password here, so use a **personal access token** instead (GitHub → Settings → Developer settings → Personal access tokens). Paste it when git asks for a password:
   ```bash
   git clone https://github.com/PSMacaroni/Ashby-.git
   cd Ashby-
   ```
4. Load Delta's Python + PyTorch, then install our extra packages (`--user` installs them in your own folder, since you can't install system-wide on Delta). Run the `source` line **in every new Delta terminal**:
   ```bash
   source scripts/delta_env.sh
   pip install --user -r requirements.txt
   ```
5. Make your private settings file and fill it in (keep the repo in your home folder `~`, which only you can read):
   ```bash
   cp .env.example .env
   nano .env        # HF_TOKEN, DATA_DIR=/projects/bfqi/shared/datasets, HF_HOME. Ctrl+O save, Ctrl+X exit
   ```
6. Check everything works:
   ```bash
   python scripts/check_setup.py
   ```

Accounts, data paths and GPU budget are in [CLAUDE.md](CLAUDE.md) under "Delta specifics". **The whole team shares 10 GPU-hours**, so run GPU work as batch jobs that end when finished (`sbatch scripts/first_look_gpu.sbatch`), not as open sessions.

**If Open OnDemand is down/slow:** use SSH instead: `ssh <netid>@login.delta.ncsa.illinois.edu` (password + Duo), then do steps 3–6 in that terminal and submit jobs with `sbatch`.

Stuck on Delta? Try the Delta docs chatbot: https://chat.illinois.edu/Delta-Documentation/chat

## Git in 5 commands

```bash
git pull                          # 1. get everyone else's latest changes (ALWAYS do this first)
git checkout -b my-feature        # 2. make your own branch to work on
git add path/to/file.py           # 3. pick the files you want to save
git commit -m "Add tile loader"   # 4. save a snapshot with a short message
git push -u origin my-feature     # 5. upload it to GitHub, then open a Pull Request
```

(GitHub Desktop has buttons for all of these if you'd rather click.)

**Rules**
- **Never commit** tokens (`.env`), slides/data, or model weights. `.gitignore` blocks the usual ones, but check what you're adding.
- **Pull first** before you start working, so you don't end up with conflicts.
- Small commits with clear messages > one giant commit at 3 a.m.

## Folder layout

```
Ashby-/
├── app/                 # Streamlit app (run: streamlit run app/streamlit_app.py)
├── src/                 # shared Python code used by notebooks, scripts, and the app
├── notebooks/           # Jupyter notebooks for experiments
├── scripts/             # command-line scripts (check_setup.py lives here)
├── data/                # local data. Git-ignored, never committed
├── .env.example         # template for your private .env
├── requirements.txt     # Python packages to install
└── README.md
```

## Team roles

| # | Role | Owns |
|---|---|---|
| 1 | **Data** | Cutting slides into tiles, quality checks (blur, background, pen marks), labels |
| 2 | **Models** | Embeddings from CONCH / MUSK / PathGen-CLIP, the tissue map, measuring accuracy |
| 3 | **Analysis + summary** | Border shape, depth of invasion, immune cells (center vs. edge), necrosis, LLM summary |
| 4 | **App + slides** | Streamlit app, hover map, summary panel, final presentation |
