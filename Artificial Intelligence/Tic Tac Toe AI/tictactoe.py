import math

def print_board(board):
    print("\n")
    for row in board:
        print(" | ".join(row))
        print("-" * 9)
    print("\n")

def check_winner(board):
    # Rows
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != ' ':
            return row[0]
    
    # Columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != ' ':
            return board[0][col]
    
    # Diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
        return board[0][2]
    
    # Tie or Game Ongoing
    if any(' ' in row for row in board):
        return None
    else:
        return 'Tie'

def minimax(board, depth, is_maximizing):
    result = check_winner(board)
    if result == 'O': # AI wins
        return 10 - depth
    elif result == 'X': # Human wins
        return depth - 10
    elif result == 'Tie':
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ' '
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ' '
                    best_score = min(score, best_score)
        return best_score

def best_move(board):
    best_score = -math.inf
    move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                score = minimax(board, 0, False)
                board[i][j] = ' '
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

def main():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    
    print("Welcome to Tic-Tac-Toe!")
    print("You are 'X'. The AI is 'O'.")
    print_board(board)
    
    while True:
        # Human Move
        while True:
            try:
                row = int(input("Enter row (0, 1, 2): "))
                col = int(input("Enter col (0, 1, 2): "))
                if 0 <= row <= 2 and 0 <= col <= 2 and board[row][col] == ' ':
                    board[row][col] = 'X'
                    break
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Enter numbers only.")
        
        print_board(board)
        
        # Check if Human won
        result = check_winner(board)
        if result:
            if result == 'Tie':
                print("It's a Tie!")
            else:
                print(f"{result} Wins!")
            break
            
        # AI Move
        print("AI is thinking...")
        move = best_move(board)
        if move:
            board[move[0]][move[1]] = 'O'
            print(f"AI chose row {move[0]}, col {move[1]}")
            print_board(board)
            
        # Check if AI won
        result = check_winner(board)
        if result:
            if result == 'Tie':
                print("It's a Tie!")
            else:
                print(f"{result} Wins!")
            break

if __name__ == "__main__":
    main()
