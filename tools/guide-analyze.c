/* guide-analyze: read guide.log, print statistics for a microsecond window.
 * Usage: guide-analyze START_US END_US [guide.log] [-v]
 * The log defaults to ~/.local/share/phi-stream/dev/guide.log; -v also
 * lists every token in the window.
 * One line per thinking token: microseconds<TAB>pos=N kl=K flip=0|1 live="TOKEN" guide="TOKEN" [experts_shared=S]
 * Output: tokens in window, mean/median KL + quartiles, flip share,
 *         top-10 live→guide changes among flips, mean experts_shared.
 * Written by the dev stream (2026-10-02); see guide-analyze.md.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_LINES 500000
#define MAX_TOKEN 64

typedef struct {
    long long t_us;
    int pos;
    double kl;
    int flip;
    char live[MAX_TOKEN];
    char guide[MAX_TOKEN];
    double experts_shared;
    int has_experts;
} Line;

static Line lines[MAX_LINES];

/* Parse one guide.log line into a Line struct.
 * Format: microseconds<TAB>pos=N\tkl=K\tflip=F\tlive="..." guide="..." [experts_shared=S]
 * The first field (before the first tab) is the raw timestamp in microseconds.
 * Remaining fields are TAB-separated key=value pairs.
 */
static void parse_line(char *buf, Line *out) {
    memset(out, 0, sizeof(*out));
    char *p = buf;

    /* First field: raw timestamp (no key= prefix). */
    char *tab1 = strchr(p, '\t');
    if (tab1) {
        *tab1 = '\0';
        out->t_us = atoll(p);
        p = tab1 + 1;
    } else {
        return; /* malformed line */
    }

    /* Remaining fields: TAB-separated key=value pairs. */
    while (p && *p) {
        char *eq = strchr(p, '=');
        if (!eq) break;
        char *key = p;
        *eq = '\0';
        char *val = eq + 1;
        p = strchr(val, '\t');
        if (p) *p++ = '\0';

        if (strcmp(key, "pos") == 0) {
            out->pos = atoi(val);
        } else if (strcmp(key, "kl") == 0) {
            out->kl = atof(val);
        } else if (strcmp(key, "flip") == 0) {
            out->flip = atoi(val);
        } else if (strcmp(key, "live") == 0) {
            /* Strip surrounding quotes */
            if (val[0] == '"') val++;
            int len = strlen(val);
            if (len > 0 && val[len-1] == '"') val[len-1] = '\0';
            strncpy(out->live, val, MAX_TOKEN-1);
        } else if (strcmp(key, "guide") == 0) {
            if (val[0] == '"') val++;
            int len = strlen(val);
            if (len > 0 && val[len-1] == '"') val[len-1] = '\0';
            strncpy(out->guide, val, MAX_TOKEN-1);
        } else if (strcmp(key, "experts_shared") == 0) {
            out->experts_shared = atof(val);
            out->has_experts = 1;
        }
    }
}

/* Read guide.log, filter by time window [start_us, end_us]. */
static int read_window(const char *path, long long start_us, long long end_us,
                       Line *out, int max_out) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char buf[4096];
    int count = 0;
    while (fgets(buf, sizeof(buf), f) && count < max_out) {
        int len = strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r'))
            buf[--len] = '\0';
        if (len == 0) continue;

        Line l;
        parse_line(buf, &l);
        if (l.t_us >= start_us && l.t_us <= end_us) {
            out[count++] = l;
        }
    }
    fclose(f);
    return count;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Sort doubles in place (qsort: an insertion sort took n^2 on a long window). */
static void sort_doubles(double *a, int n) {
    qsort(a, n, sizeof(double), cmp_double);
}

/* The q-quantile of sorted a[0..n-1], linearly interpolated (n >= 1; the
 * last element has no neighbour above it, so it is not interpolated). */
static double quantile(const double *a, int n, double q) {
    double rank = q * (n - 1);
    int lo = (int)floor(rank);
    if (lo >= n - 1) return a[n - 1];
    return a[lo] + (rank - lo) * (a[lo + 1] - a[lo]);
}

/* Compute quartiles on sorted array a[0..n-1]. */
static void quartiles(double *a, int n, double *q25, double *q50, double *q75) {
    if (n == 0) { *q25 = *q50 = *q75 = 0; return; }
    *q25 = quantile(a, n, 0.25);
    *q50 = quantile(a, n, 0.50);
    *q75 = quantile(a, n, 0.75);
}

/* Count pair occurrences, return top-10 by count. */
typedef struct { char live[MAX_TOKEN]; char guide[MAX_TOKEN]; int count; } PairCount;

static int pair_cmp(const void *a, const void *b) {
    const PairCount *pa = a, *pb = b;
    if (pa->count != pb->count) return pb->count - pa->count;
    int c = strcmp(pa->live, pb->live);
    if (c != 0) return c;
    return strcmp(pa->guide, pb->guide);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s START_US END_US [guide.log] [-v]\n", argv[0]);
        return 1;
    }
    long long start_us = atoll(argv[1]);
    long long end_us = atoll(argv[2]);
    static char def[4096];
    const char *home = getenv("HOME");
    snprintf(def, sizeof def, "%s/.local/share/phi-stream/dev/guide.log", home ? home : "");
    const char *path = def;
    int verbose = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) verbose = 1;
        else path = argv[i];
    }

    int n = read_window(path, start_us, end_us, lines, MAX_LINES);
    if (n < 0) {
        fprintf(stderr, "Cannot open %s\n", path);
        return 1;
    }
    if (n == 0) {
        printf("No tokens in window [%lld, %lld]\n", start_us, end_us);
        return 0;
    }
    if (n == MAX_LINES) {
        fprintf(stderr, "only the first %d tokens of the window are read\n", MAX_LINES);
    }

    printf("Tokens in window: %d\n", n);
    if (verbose) {
        for (int i = 0; i < n; i++) {
            printf("  pos=%d kl=%.4f flip=%d live=%s guide=%s\n",
                   lines[i].pos, lines[i].kl, lines[i].flip,
                   lines[i].live, lines[i].guide);
        }
    }

    /* KL statistics. */
    double sum_kl = 0;
    double *kl_arr = malloc(n * sizeof(double));
    int flip_count = 0;
    double experts_sum = 0;
    int experts_count = 0;

    for (int i = 0; i < n; i++) {
        kl_arr[i] = lines[i].kl;
        sum_kl += lines[i].kl;
        if (lines[i].flip) flip_count++;
        if (lines[i].has_experts) {
            experts_sum += lines[i].experts_shared;
            experts_count++;
        }
    }
    sort_doubles(kl_arr, n);
    double q25, q50, q75;
    quartiles(kl_arr, n, &q25, &q50, &q75);

    printf("\nKL statistics:\n");
    printf("  mean:     %.4f\n", sum_kl / n);
    printf("  median:   %.4f\n", q50);
    printf("  Q25:      %.4f\n", q25);
    printf("  Q75:      %.4f\n", q75);
    free(kl_arr);

    /* Flip share. */
    printf("\nFlip share: %d/%d = %.1f%%\n", flip_count, n, 100.0 * flip_count / n);

    /* Top-10 live→guide changes among flips. */
    if (flip_count > 0) {
        PairCount *pairs = calloc(flip_count, sizeof(PairCount));
        int pair_n = 0;
        for (int i = 0; i < n; i++) {
            if (!lines[i].flip) continue;
            int found = 0;
            for (int j = 0; j < pair_n; j++) {
                if (strcmp(pairs[j].live, lines[i].live) == 0 &&
                    strcmp(pairs[j].guide, lines[i].guide) == 0) {
                    pairs[j].count++;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                strncpy(pairs[pair_n].live, lines[i].live, MAX_TOKEN-1);
                strncpy(pairs[pair_n].guide, lines[i].guide, MAX_TOKEN-1);
                pairs[pair_n].count = 1;
                pair_n++;
            }
        }
        qsort(pairs, pair_n, sizeof(PairCount), pair_cmp);
        int top = pair_n < 10 ? pair_n : 10;
        printf("\nTop %d live→guide changes among flips:\n", top);
        for (int i = 0; i < top; i++) {
            printf("  %s → %s: %d\n", pairs[i].live, pairs[i].guide, pairs[i].count);
        }
        free(pairs);
    }

    /* Mean experts_shared. */
    if (experts_count > 0) {
        printf("\nMean experts_shared: %.3f (%d tokens)\n", experts_sum / experts_count, experts_count);
    } else {
        printf("\nNo experts_shared data in window.\n");
    }

    return 0;
}
