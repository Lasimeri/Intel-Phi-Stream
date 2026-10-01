/* stamp: how evenly text arrives on a pipe. Reads stdin until EOF, copies
 * it to stdout, and stamps every read with CLOCK_MONOTONIC; at the end,
 * on stderr: the reads, and the gaps between them (median, 90th, 99th
 * percentile, largest, and how many reached 100, 250 and 500 ms), after
 * skipping the first SKIP seconds after the first read (argv[1], default 0:
 * the opening, which arrives at once).
 * Built and run by tcc: `tcc -run tools/stamp.c [SKIP] < pipe`. See stamp.md. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static long long now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

static int cmp(const void *a, const void *b) {
    long long x = *(const long long *)a, y = *(const long long *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    double skip = argc > 1 ? atof(argv[1]) : 0.0;
    size_t cap = 1 << 16, n = 0;
    long long *t = malloc(cap * sizeof *t);
    char buf[65536];
    long long t0 = -1;
    for (;;) {
        ssize_t r = read(0, buf, sizeof buf);
        if (r <= 0) break;
        long long now = now_us();
        if (t0 < 0) t0 = now;
        if (fwrite(buf, 1, (size_t)r, stdout) != (size_t)r) return 1;
        fflush(stdout);
        if (now - t0 < (long long)(skip * 1e6)) continue;
        if (n == cap) {
            cap *= 2;
            t = realloc(t, cap * sizeof *t);
        }
        t[n++] = now;
    }
    if (n < 2) {
        fprintf(stderr, "stamp: %zu reads after %.1f s, no gaps\n", n, skip);
        return 0;
    }
    size_t m = n - 1;
    long long *g = malloc(m * sizeof *g);
    size_t over100 = 0, over250 = 0, over500 = 0;
    for (size_t i = 0; i < m; i++) {
        g[i] = t[i + 1] - t[i];
        over100 += g[i] >= 100000;
        over250 += g[i] >= 250000;
        over500 += g[i] >= 500000;
    }
    qsort(g, m, sizeof *g, cmp);
    fprintf(stderr,
            "stamp: %zu reads over %.1f s; gaps ms: median %.1f p90 %.1f p99 %.1f max %.1f; "
            ">=100 ms %zu, >=250 ms %zu, >=500 ms %zu\n",
            n, (t[n - 1] - t[0]) / 1e6, g[m / 2] / 1e3, g[m * 9 / 10] / 1e3,
            g[m * 99 / 100] / 1e3, g[m - 1] / 1e3, over100, over250, over500);
    return 0;
}
