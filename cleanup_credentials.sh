echo "===== REMOVE GITHUB + HUGGING FACE CREDENTIALS ====="

# GitHub credentials
rm -f ~/.git-credentials
git config --global --unset-all credential.helper 2>/dev/null || true

# Hugging Face credentials
rm -f ~/.cache/huggingface/token
rm -f ~/.huggingface/token

# Current shell environment
unset HF_TOKEN
unset HUGGINGFACE_HUB_TOKEN
unset GITHUB_TOKEN
unset GH_TOKEN

# Remove possible token exports from shell config
sed -i '/HF_TOKEN/d' ~/.bashrc ~/.profile ~/.bash_profile ~/.zshrc 2>/dev/null || true
sed -i '/HUGGINGFACE_HUB_TOKEN/d' ~/.bashrc ~/.profile ~/.bash_profile ~/.zshrc 2>/dev/null || true
sed -i '/GITHUB_TOKEN/d' ~/.bashrc ~/.profile ~/.bash_profile ~/.zshrc 2>/dev/null || true
sed -i '/GH_TOKEN/d' ~/.bashrc ~/.profile ~/.bash_profile ~/.zshrc 2>/dev/null || true

echo
echo "===== VERIFY ====="

echo "--- Git credentials ---"
if [ -f ~/.git-credentials ]; then
    echo "WARNING: ~/.git-credentials still exists"
else
    echo "OK: ~/.git-credentials removed"
fi

echo
echo "--- Hugging Face token ---"
if [ -f ~/.cache/huggingface/token ] || [ -f ~/.huggingface/token ]; then
    echo "WARNING: Hugging Face token still exists"
else
    echo "OK: Hugging Face token removed"
fi

echo
echo "--- Environment tokens ---"
if env | grep -Ei '(^|_)(HF|HUGGINGFACE|GITHUB|GH)_' >/dev/null 2>&1; then
    echo "WARNING: Possible token environment variables remain:"
    env | grep -Ei '(^|_)(HF|HUGGINGFACE|GITHUB|GH)_'
else
    echo "OK: No GitHub/HF token environment variables found"
fi

echo
echo "===== DONE ====="