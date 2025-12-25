# ❌⭕ Tic-Tac-Toe AI

**An unbeatable Tic-Tac-Toe AI implementation.**

This project features a command-line Tic-Tac-Toe game where you play against an AI agent. The AI uses the **Minimax Algorithm** (a recursive decision-making algorithm) to ensure it plays optimally. It will never lose—the best you can hope for is a draw!

## 🚀 How to Play

1.  Clone the repository.
2.  Navigate to this folder.
3.  Run the script:
    ```bash
    python tictactoe.py
    ```
4.  Follow the on-screen prompts to enter your moves (positions 0-8).

## 🧠 The Logic (Minimax)

The AI looks ahead at every possible future move.
-   If a move leads to a win, it gets a score of +10.
-   If a move leads to a loss, it gets a score of -10.
-   If a move leads to a draw, it gets a score of 0.
The AI always chooses the move with the maximum possible score, assuming you (the opponent) will also play perfectly to minimize the AI's score.
