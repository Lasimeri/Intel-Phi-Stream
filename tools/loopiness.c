/* loopiness: read a phi-stream chain.log (microseconds TAB kind TAB text),
 * take two microsecond arguments (window start, window end), and report
 * loopiness metrics for that window:
 *   - think and speak tokens, given lines
 *   - what share of 8-token think/speak sequences repeat an earlier one
 *     in the window
 *   - how many given lines contain 'from the system'
 * --kind think|speak|all (default all): only count that kind of token.
 * Build: tcc -o loopiness tools/loopiness.c
 * Run:   ./loopiness START_US END_US [--kind X] [chain.log]
 * Example: ./loopiness 1790886341549895 1790889741549895 --kind think
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_TOKENS 2000000
#define SEQ_LEN 8

static uint64_t hash_seq(const char **seq, size_t n) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < n; i++) {
        const char *s = seq[i];
        while (*s) {
            h ^= (uint64_t)(unsigned char)*s++;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

int main(int argc, char **argv) {
    const char *kind_filter = "all";
    long long start_us, end_us;
    const char *fname = "chain.log";
    int npos = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--kind") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "loopiness: --kind needs an argument\n");
                return 1;
            }
            kind_filter = argv[++i];
        } else {
            if (npos == 0) { start_us = atoll(argv[i]); npos++; }
            else if (npos == 1) { end_us = atoll(argv[i]); npos++; }
            else { fname = argv[i]; }
        }
    }
    if (npos < 2) {
        fprintf(stderr, "loopiness: START_US END_US [--kind X] [chain.log]\n");
        return 1;
    }
    /* A kind misspelled would count nothing, without a word. */
    if (strcmp(kind_filter, "all") && strcmp(kind_filter, "think") && strcmp(kind_filter, "speak")) {
        fprintf(stderr, "loopiness: --kind takes think, speak or all, not %s\n", kind_filter);
        return 1;
    }

    FILE *in = fopen(fname, "r");
    if (!in) { perror(fname); return 1; }

    char **tokens = malloc(MAX_TOKENS * sizeof *tokens);
    size_t ntoks = 0, n_think = 0, n_speak = 0;
    int n_given = 0, n_from_system = 0;
    char buf[65536];

    while (fgets(buf, sizeof buf, in)) {
        char *p = buf;
        char *tab1 = strchr(p, '\t');
        if (!tab1) continue;
        *tab1 = '\0';
        long long us = atoll(p);
        p = tab1 + 1;
        char *tab2 = strchr(p, '\t');
        if (!tab2) continue;
        *tab2 = '\0';
        char *kind = p;
        char *text = tab2 + 1;
        text[strcspn(text, "\n")] = '\0';

        if (us < start_us || us > end_us) continue;

        if (strcmp(kind, "given") == 0) {
            n_given++;
            if (strstr(text, "from the system")) n_from_system++;
        } else if (strcmp(kind, "think") == 0 || strcmp(kind, "speak") == 0) {
            if (strcmp(kind_filter, "all") != 0 && strcmp(kind_filter, kind) != 0)
                continue;
            if (ntoks >= MAX_TOKENS) break;
            if (kind[0] == 't') n_think++; else n_speak++;
            tokens[ntoks++] = strdup(text);
        }
    }
    fclose(in);

    printf("window: %.1f s (%lld..%lld us)\n",
           (end_us - start_us) / 1e6, start_us, end_us);
    printf("tokens (%s): %zu (think %zu, speak %zu)\n", kind_filter, ntoks, n_think, n_speak);

    size_t n_seqs = ntoks >= SEQ_LEN ? ntoks - SEQ_LEN + 1 : 0;
    size_t n_repeats = 0;

    /* The table at least twice the window's sequences, so the probe always
     * finds a free slot (a fixed 2^20 looped forever past a million). */
    size_t cap = 1024;
    while (cap < 2 * n_seqs) cap <<= 1;
    uint64_t *seen = calloc(cap, sizeof *seen);
    int *seen_count = calloc(cap, sizeof *seen_count);
    size_t seen_n = 0;

    for (size_t i = 0; i < n_seqs; i++) {
        uint64_t h = hash_seq((const char **)(tokens + i), SEQ_LEN);
        if (h == 0) h = 1; /* 0 marks an empty slot */
        size_t idx = h & (cap - 1);
        while (seen[idx] != 0 && seen[idx] != h) {
            idx = (idx + 1) & (cap - 1);
        }
        if (seen[idx] == 0) {
            seen[idx] = h;
            seen_count[idx] = 1;
            seen_n++;
        } else {
            seen_count[idx]++;
            n_repeats++;
        }
    }

    printf("8-token sequences: %zu\n", n_seqs);
    printf("unique sequences: %zu\n", seen_n);
    printf("repeated sequences: %zu (%.2f%%)\n",
           n_repeats, n_seqs ? 100.0 * n_repeats / n_seqs : 0.0);
    printf("given lines: %d, from the system: %d\n", n_given, n_from_system);

    free(seen);
    free(seen_count);
    for (size_t i = 0; i < ntoks; i++) free(tokens[i]);
    free(tokens);
    return 0;
}
