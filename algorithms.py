import time
import matplotlib.pyplot as plt
from collections import defaultdict

# === loading of data ===
with open("english_words.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

text = full_text.replace('\n', ' ')

# === Алгоритмы поиска подстроки ===
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
    n, m = len(text), len(pattern)
    h = pow(d, m - 1, prime)
    p = t = 0

    for i in range(m):
        p = (d * p + ord(pattern[i])) % prime
        t = (d * t + ord(text[i])) % prime

    for s in range(n - m + 1):
        if p == t:
            if text[s:s + m] == pattern:
                return s
        if s < n - m:
            t = (d * (t - ord(text[s]) * h) + ord(text[s + m])) % prime
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

# === Список алгоритмов, lists of the algorithms ===
algorithms = {
    "Brute-force": brute_force,
    "Sunday": sunday,
    "KMP": kmp,
    "FSM": fsm_search,
    "Rabin-Karp": rabin_karp,
    "Gusfield-Z": gusfield_z,
}

# === Замер времени, time meter===
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

# === Графики Part A ===
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

compare_and_plot("Gusfield-Z", "Sunday", T, P1, "Gusfield-Z vs Sunday (P = 'b')")
compare_and_plot("Rabin-Karp", "KMP", T, P2, "Rabin-Karp vs KMP (P = 'aaaa')")
compare_and_plot("Sunday", "Rabin-Karp", T, P3, "Sunday vs Rabin-Karp (P = 'a'*100)")

# === Part Two — Aunt's Namesday ===
guests = ["Alice", "Bob", "Charlie", "Daisy", "Eve", "Frank"]
dislikes = [
    ("Alice", "Bob"),
    ("Charlie", "Daisy"),
    ("Bob", "Eve"),
    ("Eve", "Frank"),
    ("Frank", "Alice")  # Not bipartite
]

graph = defaultdict(list)
for a, b in dislikes:
    graph[a].append(b)
    graph[b].append(a)

def assign_tables(graph, guests):
    color = {}
    for guest in guests:
        if guest in color:
            continue
        stack = [(guest, 0)]
        while stack:
            current, c = stack.pop()
            if current in color:
                if color[current] != c:
                    return None
                continue
            color[current] = c
            for neighbor in graph[current]:
                stack.append((neighbor, 1 - c))
    table1 = [g for g in guests if color.get(g, 0) == 0]
    table2 = [g for g in guests if color.get(g, 0) == 1]
    return table1, table2

print("\n=== Aunt's Namesday Sitting Scheme ===")
result = assign_tables(graph, guests)
if result is None:
    print("Impossible to assign guests to two tables without conflicts.")
else:
    table1, table2 = result
    print("Table 1:", table1)
    print("Table 2:", table2)


def parse_pattern(pattern):
    """Разбирает шаблон с учетом экранирования"""
    parsed = []
    i = 0
    while i < len(pattern):
        if pattern[i] == '\\':
            if i + 1 < len(pattern):
                parsed.append(('char', pattern[i + 1]))
                i += 2
            else:
                parsed.append(('char', '\\'))
                i += 1
        elif pattern[i] == '*':
            parsed.append(('star', '*'))
            i += 1
        elif pattern[i] == '?':
            parsed.append(('any', '?'))
            i += 1
        else:
            parsed.append(('char', pattern[i]))
            i += 1
    return parsed


def match_wildcard_at(text, t_idx, parsed_pattern, p_idx):
    """Проверяет, совпадает ли шаблон с подстрокой text[t_idx:]"""
    while p_idx < len(parsed_pattern):
        token_type, token_val = parsed_pattern[p_idx]
        if token_type == 'star':
            # '*' — пробуем все возможные позиции
            for skip in range(len(text) - t_idx + 1):
                if match_wildcard_at(text, t_idx + skip, parsed_pattern, p_idx + 1):
                    return True
            return False
        elif token_type == 'any':
            if t_idx >= len(text):
                return False
            t_idx += 1
            p_idx += 1
        elif token_type == 'char':
            if t_idx >= len(text) or text[t_idx] != token_val:
                return False
            t_idx += 1
            p_idx += 1
    return t_idx == len(text)


def brute_force_wildcard(text, pattern):
    """Brute-force с поддержкой ?, *, \\ """
    parsed_pattern = parse_pattern(pattern)
    for i in range(len(text)):
        if match_wildcard_at(text, i, parsed_pattern, 0):
            return True
    return False


def sunday_wildcard(text, pattern):
    """Sunday с поддержкой ?, *, \\ """
    parsed_pattern = parse_pattern(pattern)
    pattern_len_est = sum(1 for t in parsed_pattern if t[0] != 'star')
    n = len(text)
    i = 0
    while i <= n - pattern_len_est:
        if match_wildcard_at(text, i, parsed_pattern, 0):
            return True
        next_index = i + pattern_len_est
        if next_index < n:
            next_char = text[next_index]
            shift = pattern_len_est + 1
            for j in range(len(parsed_pattern) - 1, -1, -1):
                token_type, token_val = parsed_pattern[j]
                if token_type == 'char' and token_val == next_char:
                    shift = len(parsed_pattern) - j
                    break
            i += shift
        else:
            break
    return False


# === Примеры использования, Examples of use ===

tests = [
    ("hello world", "he*o", True),
    ("hello world", "he??o", False),
    ("file.txt", "file\\.*", True),
    ("abcde", "a*d?", True),
    ("abc", "a\\*c", False),
    ("abc", "a\\?c", False),
    ("a*c", "a\\*c", True),
    ("a?c", "a\\?c", True),
    ("a\\c", "a\\\\c", True),
]

print("=== Tests of Brute-force ===")
for text, pattern, expected in tests:
    result = brute_force_wildcard(text, pattern)
    print(f"Pattern: {pattern}, Text: {text} → {result} (Expected: {expected})")

print("\n=== Tests of Sunday ===")
for text, pattern, expected in tests:
    result = sunday_wildcard(text, pattern)
    print(f"Pattern: {pattern}, Text: {text} → {result} (Expected: {expected})")


# === Part Three — Jewish-style Carp ===
def hash_2d_block(block, base_row=256, base_col=257, mask=(1 << 64) - 1):
    k = len(block)
    h = 0
    for i in range(k):
        row_hash = 0
        for j in range(k):
            row_hash = ((row_hash * base_col) + hash(block[i][j])) & mask
        h = ((h * base_row) + row_hash) & mask
    return h

def get_block(matrix, x, y, k):
    return [row[y:y + k] for row in matrix[x:x + k]]

def rabin_karp_2d(matrix, k):
    if not matrix or not matrix[0] or k == 0:
        return False

    m, n = len(matrix), len(matrix[0])
    if k > m or k > n:
        return False

    pattern = [row[n - k:] for row in matrix[:k]]
    pattern_hash = hash_2d_block(pattern)

    for i in range(m - k + 1):
        for j in range(n - k + 1):
            if i == 0 and j == n - k:
                continue
            block = get_block(matrix, i, j, k)
            if hash_2d_block(block) == pattern_hash:
                if block == pattern:
                    return True
    return False

# === Проверка Carp, test Carp ===
picture = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 1, 2, 3],
    [4, 5, 6, 7],
]
K = 2
print("\n=== Jewish-style Carp ===")
print("Duplicate found:" if rabin_karp_2d(picture, K) else "No duplicate found.")


