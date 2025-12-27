# Graduation-Project

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ItsAboudTime/Graduation-Project.git
cd Graduation-Project
```

### 2. Create a Virtual Environment
- **macOS/Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **Windows (Command Prompt)**:
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  venv\Scripts\Activate.ps1
  ```

### 3. Install Python Dependencies
```bash
pip install -r dependencies.txt
```

### 4. Additional Requirements for Linux
If you are running on Linux, you must install the following system packages:

- `xdotool`
- `xrandr`

Install them using your distribution's package manager. For example, on Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install xdotool x11-xserver-utils
```

### 5. Run Examples

- CLI cursor move, click, and scroll:
```bash
python -m examples.cursor_control
```

- Linux head-controlled cursor (webcam required):
```bash
python -m examples.head_cursor
```

- Linux wink-controlled cursor (webcam required):
```bash
python -m examples.wink_cursor
```