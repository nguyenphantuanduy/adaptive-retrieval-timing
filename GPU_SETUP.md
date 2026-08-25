# ============================================================

# UPDATE + INSTALL SYSTEM DEPENDENCIES

# ============================================================

sudo apt update && \
sudo apt upgrade -y && \
sudo apt install -y \
 build-essential \
 git \
 software-properties-common \
 curl && \
sudo add-apt-repository ppa:deadsnakes/ppa -y && \
sudo apt update && \
sudo apt install -y \
 python3.13 \
 python3.13-dev \
 python3.13-venv

# ============================================================

# VERIFY PYTHON

# ============================================================

python3.13 --version

# ============================================================

# CLONE PROJECT — DEV BRANCH

# ============================================================

mkdir -p ~/projects && \
cd ~/projects && \
git clone \
 --single-branch \
 --branch dev \
 https://github.com/nguyenphantuanduy/adaptive-retrieval-timing.git

cd ~/projects/adaptive-retrieval-timing

# ============================================================

# VERIFY GIT BRANCH

# ============================================================

git branch

git status

# ============================================================

# CREATE PYTHON 3.13 ENVIRONMENT

# ============================================================

python3.13 -m venv .venv

source .venv/bin/activate

# ============================================================

# VERIFY PYTHON ENVIRONMENT

# ============================================================

python --version

which python

# ============================================================

# INSTALL PYTHON DEPENDENCIES

# ============================================================

pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio

pip install -r requirements.txt
