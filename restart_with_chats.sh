#!/bin/bash
echo "Waiting for Antigravity IDE to exit completely..."
while pgrep -f "antigravity-ide --no-sandbox" > /dev/null; do
    sleep 1
done
echo "Antigravity IDE closed. Injecting chat history into database..."
python3 /home/irbazali/Desktop/FYP/restore_fyp_chat.py
cp "$HOME/.config/Antigravity IDE/User/globalStorage/state.vscdb" "$HOME/.config/Antigravity IDE/User/globalStorage/state.vscdb.backup"
echo "Done! You can now start Antigravity IDE, and 'Aasaan Khata (FYP)' will appear in your past conversations."
