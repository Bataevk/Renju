# Renju

Gomoku game with AI opponent and 3D view.

**Current AI Status:** The current AI agent has been trained for 5,000,000 epochs and demonstrates weak performance. Future development will focus on implementing a more advanced agent, with plans for training up to 20,000,000 epochs.

## Installation

1. Clone the repository: `https://github.com/Bataevk/Renju`
2. Create a virtual environment.
3. Install the required libraries.

To install the libraries, execute the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can use:

```bash
pip install numpy icecream gymnasium torch stable-baselines3 flask flask_core
```

## Usage

1. Run the server with the command:

```bash
python .\renju_server.py
```

2. Access the application at:

```
http://127.0.0.1:5000/
```
