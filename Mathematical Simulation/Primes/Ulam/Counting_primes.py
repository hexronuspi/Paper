from PIL import Image, ImageDraw
from IPython.display import display
import os

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

def create_ulam_image(spiral, pixel_size):
    width = len(spiral[0]) * pixel_size
    height = len(spiral) * pixel_size
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for y, row in enumerate(spiral):
        for x, num in enumerate(row):
            if num == 1:
                draw.rectangle([(x * pixel_size, y * pixel_size),
                                ((x + 1) * pixel_size, (y + 1) * pixel_size)], fill="black")
            elif is_prime(num):
                draw.rectangle([(x * pixel_size, y * pixel_size),
                                ((x + 1) * pixel_size, (y + 1) * pixel_size)], fill="red")

    return image

def draw_circle(draw, center_x, center_y, radius):
    draw.ellipse([(center_x - radius, center_y - radius),
                  (center_x + radius, center_y + radius)], outline="blue")

def count_numbers_in_circle(spiral, center_x, center_y, radius):
    count_primes = 0
    count_composites = 0

    for y in range(len(spiral)):
        for x in range(len(spiral[0])):
            if (x - center_x) ** 2 + (y - center_y) ** 2 < radius ** 2:
                if spiral[y][x] != 0:
                    if is_prime(spiral[y][x]):
                        count_primes += 1
                    else:
                        count_composites += 1

    return count_primes, count_composites

n = 111  # Adjust this value to change the size of the spiral (use an odd number)
pixel_size = 5  # Size of each pixel in the image (adjust as needed)
radius = 30 # Adjust this value to change the radius of the circle

ulam = ulam_spiral(n)
ulam_image = create_ulam_image(ulam, pixel_size)
draw = ImageDraw.Draw(ulam_image)
center_x, center_y = n // 2, n // 2
draw_circle(draw, center_x * pixel_size, center_y * pixel_size, radius * pixel_size)
ulam_image.save("ulam_spiral_with_circle.png")

# Display the image in Google Colab
display(ulam_image)

primes_inside_circle, composites_inside_circle = count_numbers_in_circle(ulam, center_x, center_y, radius)
total_numbers_displayed = n * n
print(f"Total natural numbers displayed: {total_numbers_displayed}")
print(f"Primes inside circle: {primes_inside_circle}")
print(f"Composites inside circle: {composites_inside_circle}")
