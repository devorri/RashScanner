import os
import sys
import shutil
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Priority logo sourcing (Check root logo.png first)
root_logo = os.path.join(BASE_DIR, "logo.png")
artifact_logo = r"C:\Users\Tom Pc\.gemini\antigravity-ide\brain\b910e55c-2f05-41ca-982f-29cdc80f42f9\app_logo_1789183970633.jpg"

source_logo = None
if os.path.exists(root_logo):
    source_logo = root_logo
elif os.path.exists(artifact_logo):
    source_logo = artifact_logo

logo_png = os.path.join(STATIC_DIR, "app_logo.png")
logo_ico = os.path.join(STATIC_DIR, "app_logo.ico")

if source_logo and os.path.exists(source_logo):
    try:
        shutil.copyfile(source_logo, logo_png)
        shutil.copyfile(source_logo, os.path.join(STATIC_DIR, "logo.png"))
        print(f"[Shortcuts] Using custom user logo from '{source_logo}' -> static/app_logo.png")
    except Exception as e:
        print(f"[Shortcuts Note] Logo copy: {e}")

# Generate Windows ICO file if PIL is available
if os.path.exists(logo_png):
    try:
        from PIL import Image
        img = Image.open(logo_png)
        img.save(logo_ico, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
        print(f"[Shortcuts] Created ICO icon: {logo_ico}")
    except Exception as e:
        print(f"[Shortcuts Note] Could not generate .ico: {e}")

# Remove old legacy files if present
old_files_to_clean = [
    "start_kiosk_current.sh", "start_kiosk_rash22.sh", "start_kiosk_rash50.sh",
    "start_current.bat", "start_rash22.bat", "start_rash50.bat",
    "RashScanner_Current.desktop", "RashScanner_Rash22.desktop", "RashScanner_Rash50.desktop"
]

for f in old_files_to_clean:
    p = os.path.join(BASE_DIR, f)
    if os.path.exists(p):
        try:
            os.remove(p)
        except Exception:
            pass

# Create single unified Windows launcher start.bat
start_bat_content = f"""@echo off
title Rashilience 100% AI Scanner
cd /d "{BASE_DIR}"
start http://localhost:5000
python app.py
pause
"""
with open(os.path.join(BASE_DIR, "start.bat"), "w", encoding="utf-8", newline="\r\n") as f:
    f.write(start_bat_content)

# Linux / Raspberry Pi OS .desktop shortcut
unix_base = BASE_DIR.replace("\\", "/")
unix_icon = logo_png.replace("\\", "/") if os.path.exists(logo_png) else "utilities-terminal"

desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Rashilience Scanner
Comment=Launch 100% AI Skin Disease Decision Support System & Kiosk
Exec=/bin/bash -c "cd '{unix_base}' && ./start_kiosk.sh"
Icon={unix_icon}
Terminal=true
Categories=Medical;Science;Education;AI;
Path={unix_base}
StartupNotify=true
"""

desktop_filepath = os.path.join(BASE_DIR, "RashScanner.desktop")
with open(desktop_filepath, "w", encoding="utf-8", newline="\n") as f:
    f.write(desktop_content)
print(f"[Shortcuts] Created unified desktop file: RashScanner.desktop")

# Install to Desktop directory (~/Desktop or Windows Desktop)
home_dir = os.path.expanduser("~")
desktop_dir = os.path.join(home_dir, "Desktop")
app_menu_dir = os.path.join(home_dir, ".local", "share", "applications")

if os.path.exists(desktop_dir):
    # Remove old .desktop files from ~/Desktop
    for f in old_files_to_clean:
        if f.endswith(".desktop"):
            old_dst = os.path.join(desktop_dir, f)
            if os.path.exists(old_dst):
                try:
                    os.remove(old_dst)
                except Exception:
                    pass

    # 1. Linux / Pi Desktop Icon
    dst = os.path.join(desktop_dir, "RashScanner.desktop")
    try:
        shutil.copyfile(desktop_filepath, dst)
        os.chmod(dst, 0o755)
        print(f"[Shortcuts] Installed app icon on Raspberry Pi Desktop: {dst}")
    except Exception as e:
        print(f"[Shortcuts Note] Desktop copy note: {e}")

    # 2. Windows Desktop LNK Shortcut with Custom Icon
    if sys.platform.startswith("win"):
        # Remove old .bat shortcut if present
        old_bat = os.path.join(desktop_dir, "Rashilience Scanner.bat")
        if os.path.exists(old_bat):
            try:
                os.remove(old_bat)
            except Exception:
                pass

        # Create Windows .lnk shortcut with custom icon via PowerShell
        lnk_path = os.path.join(desktop_dir, "Rashilience Scanner.lnk")
        target_cmd = "cmd.exe"
        start_bat_path = os.path.join(BASE_DIR, "start.bat")
        cmd_args = f'/c "{start_bat_path}"'
        icon_loc = logo_ico if os.path.exists(logo_ico) else logo_png

        ps_command = f'''
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{lnk_path}')
$Shortcut.TargetPath = '{target_cmd}'
$Shortcut.Arguments = '{cmd_args}'
$Shortcut.WorkingDirectory = '{BASE_DIR}'
if ("{icon_loc}" -and (Test-Path "{icon_loc}")) {{
    $Shortcut.IconLocation = '{icon_loc}'
}}
$Shortcut.Save()
'''
        ps_file = os.path.join(BASE_DIR, "_temp_shortcut.ps1")
        try:
            with open(ps_file, "w", encoding="utf-8") as f:
                f.write(ps_command)
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", ps_file], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(ps_file):
                os.remove(ps_file)
            print(f"[Shortcuts] Created Windows Desktop Icon Shortcut (.lnk) with custom logo: {lnk_path}")
        except Exception as e:
            print(f"[Shortcuts Note] PowerShell shortcut generation note: {e}")

if os.path.exists(app_menu_dir):
    dst = os.path.join(app_menu_dir, "RashScanner.desktop")
    try:
        shutil.copyfile(desktop_filepath, dst)
        os.chmod(dst, 0o755)
        print(f"[Shortcuts] Installed app icon in Raspberry Pi App Menu: {dst}")
    except Exception as e:
        pass
