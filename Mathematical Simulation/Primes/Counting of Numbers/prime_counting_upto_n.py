def generate_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        for p in primes:
            if num % p == 0:
                break
        else:
            primes.append(num)
        num += 1
    return primes

prime_list = generate_primes(5000)

for i in range(len(prime_list)):
    print(prime_list[i], end=", ")
