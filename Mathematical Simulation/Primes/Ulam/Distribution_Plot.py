def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

def ulam_spiral(n):
    spiral = [[0] * n for _ in range(n)]
    x, y = n // 2, n // 2
    spiral[x][y] = 1

    dx, dy = 1, 0
    length = 1
    num = 2

    while length < n:
        for _ in range(2):
            for _ in range(length):
                x, y = x + dx, y + dy
                if num <= n * n:
                    spiral[x][y] = num
                num += 1
            dx, dy = -dy, dx
        length += 1

    return spiral

def print_ulam_spiral(spiral):
    for row in spiral:
        for num in row:
            if num == 0:
                print("   ", end=" ")
            elif is_prime(num):
                print(f"\033[91m{num:3}\033[0m", end=" ")
            else:
                print(f"\033[30m{num:3}\033[0m", end=" ")
        print()

n = 21  # Adjust this value to change the size of the spiral
ulam = ulam_spiral(n)
print_ulam_spiral(ulam)
