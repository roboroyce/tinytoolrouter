#!/bin/bash
# TinyToolRouter remote pipeline. Runs on the temporary AWS worker via user-data.
# Expects BUCKET env (S3 bucket holding project.tar.gz). Self-terminates.
set -uo pipefail

LOG=/var/log/tinytoolrouter.log
mkdir -p /var/log
exec >>"$LOG" 2>&1
echo "=== bootstrap start $(date -u +%FT%TZ) ==="

PROJ=/opt/proj
VENV=/opt/venv
BUCKET="${BUCKET:?BUCKET not set}"
export HF_HOME=/opt/hf
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# ---------------------------------------------------------------- metadata
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 600")
IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/placement/region)
ITYPE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/instance-type)
echo "instance=$IID type=$ITYPE region=$REGION"
START_EPOCH=$(date +%s)

aws s3 cp "$LOG" "s3://$BUCKET/logs/bootstrap-start.log" --region "$REGION" || true

# ------------------------------------------------------- deadline + shutdown
# Independent in-instance timer; poweroff triggers Terminate because the
# instance was launched with --instance-initiated-shutdown-behavior terminate.
cat >/etc/systemd/system/worker-deadline.service <<'EOF'
[Unit]
Description=Worker deadline poweroff
[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl poweroff
EOF
cat >/etc/systemd/system/worker-deadline.timer <<'EOF'
[Unit]
Description=Worker deadline timer
[Timer]
OnActiveSec=168min
AccuracySec=30s
Unit=worker-deadline.service
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now worker-deadline.timer
systemctl list-timers worker-deadline.timer --no-pager || true
echo "=== deadline timer configured ==="

finish() {
  rc=$?
  END_EPOCH=$(date +%s); RUNTIME=$((END_EPOCH-START_EPOCH))
  echo "=== bootstrap end rc=$rc runtime=${RUNTIME}s $(date -u +%FT%TZ) ==="
  {
    echo "instance_id=$IID"
    echo "instance_type=$ITYPE"
    echo "region=$REGION"
    echo "exit_code=$rc"
    echo "runtime_seconds=$RUNTIME"
    echo "finished_at=$(date -u +%FT%TZ)"
  } >/tmp/summary.txt
  aws s3 cp "$LOG" "s3://$BUCKET/logs/tinytoolrouter.log" --region "$REGION" || true
  aws s3 cp /tmp/summary.txt "s3://$BUCKET/logs/summary.txt" --region "$REGION" || true
  if [ "$rc" -eq 0 ]; then
    echo ok >/tmp/DONE && aws s3 cp /tmp/DONE "s3://$BUCKET/logs/DONE" --region "$REGION" || true
  else
    echo "failed:$rc" >/tmp/FAILED && aws s3 cp /tmp/FAILED "s3://$BUCKET/logs/FAILED" --region "$REGION" || true
  fi
  sync
  /usr/bin/systemctl poweroff
}
trap finish EXIT

# ------------------------------------------------------------- project code
mkdir -p "$PROJ" && cd "$PROJ"
aws s3 cp "s3://$BUCKET/project.tar.gz" project.tar.gz --region "$REGION"
tar xzf project.tar.gz
cd "$PROJ/project"
ls -la
echo "=== python ==="; python3 --version; nvidia-smi || echo "no nvidia-smi"

# ----------------------------------------------------------- python packages
python3 -m venv "$VENV"
. "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers accelerate peft datasets "huggingface_hub<1.0"
python3 - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__, "peft", peft.__version__)
PY

# --------------------------------------------------------------- 1) baseline
echo "=== baseline eval: Qwen/Qwen3-0.6B ==="
python3 src/eval_harness.py --data data/test.jsonl --backend transformers \
    --model Qwen/Qwen3-0.6B --out reports/base.json || echo "baseline eval failed"

# ------------------------------------------------------------------ 2) train
MERGED=artifacts/tinytoolrouter-0.6b/merged
echo "=== LoRA training ==="
if python3 src/train_lora.py --data data --out artifacts/tinytoolrouter-0.6b \
    --base Qwen/Qwen3-0.6B --seed 7 --epochs 3 --merge; then
  echo "training OK"
else
  echo "training failed"
fi
ls -la artifacts/tinytoolrouter-0.6b 2>/dev/null || true

# ------------------------------------------------------------ 3) finetuned eval
if [ -d "$MERGED" ]; then
  echo "=== finetuned eval ==="
  python3 src/eval_harness.py --data data/test.jsonl --backend transformers \
      --model "$MERGED" --out reports/finetune.json || echo "finetune eval failed"
else
  echo "=== finetuned eval SKIPPED: $MERGED missing ==="
fi

# ------------------------------------------------------------------ 4) GGUF
echo "=== GGUF export (best effort) ==="
if [ -d "$MERGED" ]; then
  timeout 1500 bash scripts/remote/make_gguf.sh "$MERGED" artifacts/gguf \
      || echo "gguf export failed"
else
  echo "gguf export skipped"
fi

# ---------------------------------------------------------------- 5) upload
echo "=== uploading artifacts ==="
aws s3 cp reports/base.json "s3://$BUCKET/artifacts/reports/base.json" --region "$REGION" || true
aws s3 cp reports/finetune.json "s3://$BUCKET/artifacts/reports/finetune.json" --region "$REGION" || true
aws s3 cp data/MANIFEST.json "s3://$BUCKET/artifacts/MANIFEST.json" --region "$REGION" || true
if [ -d artifacts/tinytoolrouter-0.6b ]; then
  aws s3 sync artifacts/tinytoolrouter-0.6b/merged \
      "s3://$BUCKET/artifacts/merged" --region "$REGION" || true
  aws s3 cp artifacts/tinytoolrouter-0.6b/run_meta.json \
      "s3://$BUCKET/artifacts/run_meta.json" --region "$REGION" || true
fi
if [ -d artifacts/gguf ]; then
  aws s3 sync artifacts/gguf "s3://$BUCKET/artifacts/gguf" --region "$REGION" || true
fi
echo "=== pipeline complete $(date -u +%FT%TZ) ==="