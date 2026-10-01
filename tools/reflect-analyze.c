/* reflect-analyze.c -- parse per-token retention log and print a
   top1p histogram (5% buckets 0..95%) plus changed/kept counts and
   rejection rate (keep < WRITE_THRESHOLD). Build: tcc -o reflect-analyze reflect-analyze.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WRITE_THRESHOLD 0.45

/* Claude's: a table of 5 percent buckets, the empty ones left out. */
static void table(const char *name, const int *b) {
    printf("%s:\n", name);
    for (int i = 0; i < 20; i++) if (b[i] > 0) printf("[%2d-%2d%%]=%d\n", i*5, i*5+4, b[i]);
}

/* Claude's: a value's bucket. */
static int bucket(double v) {
    int b = (int)(v * 20);
    if (b >= 20) b = 19;
    if (b < 0) b = 0;
    return b;
}

int main(int argc, char **argv) {
    int buckets[20] = {0};
    int keeps[20] = {0}, fmts[20] = {0}; /* Claude's: keep= and fmt= */
    char outcomes[8][16]; int n_outcome[8] = {0}, n_outcomes = 0; /* Claude's: every outcome */
    int n_changed = 0, n_kept = 0;
    char buf[4096];
    int total = 0;
    int rejected = 0; // tokens where keep < THRESHOLD
    int accepted = 0;
    FILE *in = stdin;

    /* Claude's: the log as an argument too (an argument was ignored and stdin read). */
    if (argc > 1 && !(in = fopen(argv[1], "r"))) { perror(argv[1]); return 1; }
    while (fgets(buf, sizeof(buf), in)) {
        char *p = buf;
        while (*p && *p != '\n') {
            while (*p == ' ') p++;
            if (!*p || *p == '\n') break;
            char *eq = strchr(p, '=');
            if (!eq) break;
            *eq = '\0';
            char *key = p;
            char *val = eq + 1;
            /* terminate val at the next space so it is a clean C string */
            char *sp = strchr(val, ' ');
            if (sp) *sp = '\0';
            /* Claude's: the line's own newline is not part of the last value */
            val[strcspn(val, "\n")] = '\0';

            if (strcmp(key, "top1p") == 0) {
                buckets[bucket(atof(val))]++;
            }
            if (strcmp(key, "outcome") == 0) {
                if (strcmp(val, "changed") == 0) n_changed++;
                else if (strcmp(val, "kept") == 0) n_kept++; /* only kept (Claude) */
                int i = 0;
                while (i < n_outcomes && strcmp(outcomes[i], val)) i++;
                if (i == n_outcomes && n_outcomes < 8) snprintf(outcomes[n_outcomes++], 16, "%s", val);
                if (i < n_outcomes) n_outcome[i]++;
            }
            if (strcmp(key, "keep") == 0) {
                double kv = atof(val);
                if (kv < WRITE_THRESHOLD) rejected++;
                else accepted++;
                keeps[bucket(kv)]++;
            }
            if (strcmp(key, "fmt") == 0) fmts[bucket(atof(val))]++;
            p = sp ? sp + 1 : val + strlen(val); /* the fix it diagnosed */
        }
        total++;
    }

    printf("total=%d changed=%d kept=%d rejected_at_threshold=%.2f(%d)\n",
           total, n_changed, n_kept, (double)WRITE_THRESHOLD, rejected);
    printf("below_threshold_0.45=%d (%.1f%%)\n", rejected, total ? 100.0*rejected/total : 0.0);
    /* Claude's from here: every outcome, then the three tables. */
    printf("outcomes:");
    for (int i = 0; i < n_outcomes; i++) printf(" %s=%d", outcomes[i], n_outcome[i]);
    printf("\n");
    table("top1p (the likeliest token's share of everything)", buckets);
    table("keep (keep's share of keep plus write)", keeps);
    table("fmt (keep plus write: how much answered at all)", fmts);
    return 0;
}
