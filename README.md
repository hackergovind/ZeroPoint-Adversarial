# ZeroPoint: Automated AI Red Teaming Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ZeroPoint** is a research-grade, automated "Red Teaming" agent designed to assess the robustness of Artificial Intelligence systems. It goes beyond static scanning by actively planning and executing adversarial attacks against both **Computer Vision** models and **Large Language Models (LLMs)**.

> **⚠️ Educational Notice**: This tool includes real adversarial attack implementations (FGSM, Prompt Injection). It is intended solely for security research, QA testing, and educational purposes on systems you own or have explicit permission to test.

## 🚀 Features

*   **Visual Adversarial Attacks**: Implements the **Fast Gradient Sign Method (FGSM)** to generate adversarial examples that fool ResNet/CNN classifiers while remaining visually imperceptible to humans.
*   **Automated Prompt Injection**: Equipped with a library of dynamic "Jailbreak" payloads (e.g., DAN, System Override) to bypass LLM safety filters and extract system prompts.
*   **Transferability Testing**: Demonstrates black-box transfer attacks where examples generated on a local surrogate model successfully fool a remote target.
*   **Live Target Environment ("Glass-Jaw")**: Includes a fully functional, vulnerable FastAPI application (hosting ResNet-18 and a RAG simulation) for safe, legal "live fire" exercises.

## 🛠️ Architecture

The framework consists of two main components:

1.  **The Agent (`/agent`)**: The offensive core. It orchestrates attacks, manages payloads, and interprets target responses.
2.  **The Target (`/targets`)**: A compliant, vulnerable environment ("Glass-Jaw") designed to demonstrate the attacks safely.

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/hackergovind/ZeroPoint-Adversarial.git
    cd ZeroPoint-Adversarial
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## ⚔️ Usage (Live Fire Exercise)

To see ZeroPoint in action, you will run both the **Target** and the **Attacker** locally.

### Step 1: Start the Vulnerable Target
Open a terminal and launch the "Glass-Jaw" application. This simulates a production AI API.
```bash
python targets/glass_jaw_app.py
```
*Output: `Uvicorn running on http://127.0.0.1:8000`*

### Step 2: Launch the Attack
Open a **second terminal** and run the agent.
```bash
python agent/core.py --mode all
```

### Expected Output
1.  **Visual Attack**: The agent downloads a sample "Panda" image, applies gradient noise, and successfully tricks the target into classifying it as a "Staffordshire Bullterrier" (or similar) with high confidence.
2.  **Text Attack**: The agent cycles through injection payloads until it bypasses the Chatbot's safety protocols and extracts the hidden flag: `FLAG-{ZERO_POINT_ADVERSARIAL_SUCCESS}`.

## 🛡️ Defense & Mitigation
This tool highlights the need for:
*   **Adversarial Training**: Including adversarial examples in the training dataset.
*   **Input Sanitization**: rigorous checking of text inputs for injection patterns.
*   **Robustness Certification**: Verifying model behavior under perturbation.

## 📄 License
MIT License - see [LICENSE](LICENSE) for details.
