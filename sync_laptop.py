import os
import subprocess
import datetime
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

def run(cmd, check=True, capture=False):
    result = subprocess.run(
        cmd,
        cwd=REPO_DIR,
        shell=True,
        text=True,
        capture_output=capture
    )
    if check and result.returncode != 0:
        print("❌ خطا در اجرا:", cmd)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        sys.exit(result.returncode)
    return result

def has_changes():
    r = run("git status --porcelain", capture=True)
    return bool(r.stdout.strip())

if __name__ == "__main__":
    print("📂 مسیر ریپو:", REPO_DIR)

    # 1) اول آخرین تغییرات از GitHub بگیر تا با VPS هماهنگ باشه
    print("⬇️  git pull --rebase origin main")
    run("git pull --rebase origin main", check=False)

    # 2) اگر تغییری برای ارسال نیست، خروج
    if not has_changes():
        print("ℹ️ هیچ تغییری برای push وجود ندارد.")
        sys.exit(0)

    # 3) اگر هست، commit + push خودکار
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = f'auto: sync from laptop {ts}'

    print("➕ git add .")
    run("git add .")

    print(f"📝 git commit -m \"{msg}\"")
    run(f'git commit -m "{msg}"', check=False)

    print("⬆️  git push origin main")
    run("git push origin main")

    print("✅ لپ‌تاپ → GitHub با موفقیت سینک شد.")
