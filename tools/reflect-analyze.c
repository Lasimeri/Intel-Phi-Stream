/* reflect-analyze.c -- parse per-token retention log and print a
   top1p histogram (5% buckets 0..95%) plus changed/kept counts and
   rejection rate (keep < WRITE_THRESHOLD). Build: tcc -o reflect-analyze reflect-analyze.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WRITE_THRESHOLD 0.45

int main(void) {
    int buckets[20] = {0};
    int n_changed = 0, n_kept = 0;
    char buf[4096];
    int total = 0;
    int rejected = 0; // tokens where keep < THRESHOLD
    int accepted = 0;

    while (fgets(buf, sizeof(buf), stdin)) {
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

            if (strcmp(key, "top1p") == 0) {
                double v = atof(val);
                int b = (int)(v * 20);
                if (b >= 20) b = 19;
                if (b < 0) b = 0;
                buckets[b]++;
            }
            if (strcmp(key, "outcome") == 0) {
                if (strcmp(val, "changed") == 0) n_changed++;
                else if (strcmp(val, "kept") == 0) n_kept++; /* only kept (Claude) */
            }
            if (strcmp(key, "keep") == 0) {
                double kv = atof(val);
                if (kv < WRITE_THRESHOLD) rejected++;
                else accepted++;
            }
            p = sp ? sp + 1 : val + strlen(val); /* the fix it diagnosed */
        }
        total++;
    }

    printf("total=%d changed=%d kept=%d rejected_at_threshold=%.2f(%d)\n",
           total, n_changed, n_kept, (double)WRITE_THRESHOLD, rejected);
    printf("below_threshold_0.45=%d (%.1f%%)\n", rejected, total ? 100.0*rejected/total : 0.0);
    /* the next three lines are Claude's: its block was cut here by its own ```md fence */
    for (int i = 0; i < 20; i++) if (buckets[i] > 0) printf("[%2d-%2d%%]=%d\n", i*5, i*5+4, buckets[i]);
    return 0;
}
