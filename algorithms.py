import time
import matplotlib.pyplot as plt

# Загрузка текста
with open("english_words.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

text = full_text.replace('\n', ' ')

# Алгоритмы
def brute_force(text, pattern):
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            return i
    return -1

def sunday(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    shift = {pattern[i]: m - i for i in range(m)}
    i = 0
    while i <= n - m:
        if text[i:i + m] == pattern:
            return i
        next_char_index = i + m
        if next_char_index < n:
            i += shift.get(text[next_char_index], m + 1)
        else:
            break
    return -1

def kmp(text, pattern):
    def build_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    n, m = len(text), len(pattern)
    lps = build_lps(pattern)
    i = j = 0
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == m:
            return i - j
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1

def build_fsm(pattern):
    alphabet = set(pattern)
    m = len(pattern)
    fsm = [{} for _ in range(m + 1)]
    for state in range(m + 1):
        for char in alphabet:
            next_state = min(m, state + 1)
            while not pattern[:state] + char == pattern[:next_state]:
                next_state -= 1
                if next_state == 0:
                    break
            fsm[state][char] = next_state
    return fsm

def fsm_search(text, pattern):
    fsm = build_fsm(pattern)
    state = 0
    for i, char in enumerate(text):
        state = fsm[state].get(char, 0)
        if state == len(pattern):
            return i - len(pattern) + 1
    return -1

def rabin_karp(text, pattern, prime=101):
    d = 256
    n = len(text)
    m = len(pattern)
    h = pow(d, m-1) % prime
    p = t = 0

    for i in range(m):
        p = (d * p + ord(pattern[i])) % prime
        t = (d * t + ord(text[i])) % prime

    for s in range(n - m + 1):
        if p == t:
            if text[s:s + m] == pattern:
                return s
        if s < n - m:
            t = (d*(t - ord(text[s])*h) + ord(text[s + m])) % prime
            if t < 0:
                t += prime
    return -1

def compute_z(s):
    z = [0] * len(s)
    l = r = 0
    for i in range(1, len(s)):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < len(s) and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    return z

def gusfield_z(text, pattern):
    concat = pattern + "$" + text
    z = compute_z(concat)
    m = len(pattern)
    for i in range(len(z)):
        if z[i] == m:
            return i - m - 1
    return -1

# Алгоритмы
algorithms = {
    "Brute-force": brute_force,
    "Sunday": sunday,
    "KMP": kmp,
    "FSM": fsm_search,
    "Rabin-Karp": rabin_karp,
    "Gusfield-Z": gusfield_z,
}

# Измерение времени
def measure_time(algorithm, text, pattern):
    start = time.perf_counter()
    algorithm(text, pattern)
    return time.perf_counter() - start

# === Part A ===
short_pattern = "abandon abandoned abandoner"
long_pattern = " ".join(text.split()[1000:1050])
lengths = [10_000, 20_000, 40_000, 80_000, 160_000]

results_short = {name: [] for name in algorithms}
results_long = {name: [] for name in algorithms}

for length in lengths:
    t = text[:length]
    for name, algo in algorithms.items():
        results_short[name].append(measure_time(algo, t, short_pattern))
        results_long[name].append(measure_time(algo, t, long_pattern))

# Графики для Part A
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
for name in algorithms:
    axs[0].plot(lengths, results_short[name], label=name)
    axs[1].plot(lengths, results_long[name], label=name)

axs[0].set_title("Short Pattern")
axs[0].set_xlabel("Text Length (chars)")
axs[0].set_ylabel("Execution Time (s)")
axs[0].grid(True)
axs[0].legend()

axs[1].set_title("Long Pattern")
axs[1].set_xlabel("Text Length (chars)")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

# === Part B ===

# Входы
T = "a" * 100_000 + "b"
P1 = "b"
P2 = "aaaa"
P3 = "a" * 100

def compare_and_plot(name1, name2, text, pattern, title):
    t1 = measure_time(algorithms[name1], text, pattern)
    t2 = measure_time(algorithms[name2], text, pattern)
    ratio = round(t1 / t2, 2) if t2 != 0 else float('inf')

    print(f"\n{name1} vs {name2} on pattern '{pattern[:30]}...'")
    print(f"{name1}: {t1:.6f}s")
    print(f"{name2}: {t2:.6f}s")
    print(f"Speed ratio {name1}/{name2} = {ratio}")

    plt.figure(figsize=(6, 4))
    plt.bar([name1, name2], [t1, t2], color=['#1f77b4', '#ff7f0e'])
    plt.title(f"{title}\nRatio: {ratio}")
    plt.ylabel("Execution Time (s)")
    plt.grid(True, axis='y')
    plt.show()

# Sunday vs Gusfield-Z
compare_and_plot("Gusfield-Z", "Sunday", T, P1, "Gusfield-Z vs Sunday (P = 'b')")

# KMP vs Rabin-Karp
compare_and_plot("Rabin-Karp", "KMP", T, P2, "Rabin-Karp vs KMP (P = 'aaaa')")

# Rabin-Karp vs Sunday
compare_and_plot("Sunday", "Rabin-Karp", T, P3, "Sunday vs Rabin-Karp (P = 'a'*100)")
