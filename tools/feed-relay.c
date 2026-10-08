/* feed-relay: the camera and microphone feeds of one machine kept in the
 * workspace of a service on another (the harness on the GPU rack, its
 * feeds written on the desktop). Two halves joined by a pipe, usually ssh:
 *
 *   feed-relay send DIR | ssh HOST feed-relay recv DIR
 *
 * `send` follows every NAME that has a NAME.status in DIR: the status file
 * (rewritten by its writer each second) is sent whole when it changes, the
 * log NAME.log as whole lines appended since the last look, from its last
 * 64 KiB when first seen, from its start again when it shrinks or is
 * replaced (a rotation). A file with no .status beside it (events.log, the
 * service's own) is left alone. `recv` applies each frame to its DIR: a
 * status by a temporary file and a rename, a log by appending, a reset by
 * truncating. See feed-relay.md. Built by tcc: `tcc -o feed-relay tools/feed-relay.c`. */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAXF 64
#define NAMELEN 120
#define FIRST_TAIL 65536
#define CHUNK 262144

/* One followed name: its status as last sent, its log's identity and how
 * far into it the whole lines have been sent. */
struct feed {
    char name[NAMELEN];
    long long st_mtime_ns, st_size;
    unsigned long long log_ino;
    long long log_off;
    int log_seen;
};

static struct feed feeds[MAXF];
static int nfeeds;
static char buf[CHUNK];

static int good_name(const char *n) {
    size_t l = strlen(n);
    if (l == 0 || l >= NAMELEN || n[0] == '.')
        return 0;
    for (size_t i = 0; i < l; i++)
        if (n[i] == '/' || n[i] == ' ' || n[i] == '\n')
            return 0;
    return 1;
}

static struct feed *feed_of(const char *name) {
    for (int i = 0; i < nfeeds; i++)
        if (!strcmp(feeds[i].name, name))
            return &feeds[i];
    if (nfeeds == MAXF)
        return NULL;
    struct feed *f = &feeds[nfeeds++];
    memset(f, 0, sizeof *f);
    strcpy(f->name, name);
    f->st_mtime_ns = -1;
    return f;
}

/* A frame: the header line, then LEN bytes. A failed write is the far end
 * gone: exit, so whatever runs this starts it again. */
static void frame(char kind, const char *file, const char *data, long long len) {
    if (printf("%c %s %lld\n", kind, file, len) < 0 || (len && fwrite(data, 1, len, stdout) != (size_t)len)) {
        fprintf(stderr, "feed-relay: the pipe closed\n");
        exit(1);
    }
}

static void send_status(const char *dir, struct feed *f) {
    char path[4096];
    struct stat st;
    snprintf(path, sizeof path, "%s/%s.status", dir, f->name);
    if (stat(path, &st) != 0)
        return;
    long long m = (long long)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec;
    if (m == f->st_mtime_ns && st.st_size == f->st_size)
        return;
    FILE *in = fopen(path, "rb");
    if (!in)
        return;
    size_t n = fread(buf, 1, sizeof buf, in);
    fclose(in);
    char file[NAMELEN + 8];
    snprintf(file, sizeof file, "%s.status", f->name);
    frame('S', file, buf, (long long)n);
    f->st_mtime_ns = m;
    f->st_size = st.st_size;
}

static void send_log(const char *dir, struct feed *f) {
    char path[4096], file[NAMELEN + 8];
    struct stat st;
    snprintf(path, sizeof path, "%s/%s.log", dir, f->name);
    snprintf(file, sizeof file, "%s.log", f->name);
    if (stat(path, &st) != 0)
        return;
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return;
    if (!f->log_seen || st.st_ino != f->log_ino || st.st_size < f->log_off) {
        /* First sight: the last FIRST_TAIL bytes from a line's start;
         * a new or shrunk file: from its start. */
        long long from = 0;
        if (!f->log_seen && st.st_size > FIRST_TAIL) {
            from = st.st_size - FIRST_TAIL;
            if (pread(fd, buf, 4096, from) > 0) {
                char *nl = memchr(buf, '\n', 4096);
                from += nl ? (nl - buf) + 1 : 0;
            }
        }
        frame('T', file, NULL, 0);
        f->log_seen = 1;
        f->log_ino = st.st_ino;
        f->log_off = from;
    }
    while (f->log_off < st.st_size) {
        long long want = st.st_size - f->log_off;
        if (want > CHUNK)
            want = CHUNK;
        ssize_t n = pread(fd, buf, (size_t)want, f->log_off);
        if (n <= 0)
            break;
        /* Whole lines only: a line being written goes next time. */
        ssize_t keep = n;
        while (keep > 0 && buf[keep - 1] != '\n')
            keep--;
        if (keep == 0)
            break;
        frame('A', file, buf, keep);
        f->log_off += keep;
    }
    close(fd);
}

static int sender(const char *dir) {
    signal(SIGPIPE, SIG_IGN);
    for (;;) {
        DIR *d = opendir(dir);
        if (!d) {
            fprintf(stderr, "feed-relay: %s: %s\n", dir, strerror(errno));
            return 1;
        }
        struct dirent *e;
        while ((e = readdir(d))) {
            size_t l = strlen(e->d_name);
            if (l <= 7 || strcmp(e->d_name + l - 7, ".status"))
                continue;
            char name[NAMELEN];
            if (l - 7 >= NAMELEN)
                continue;
            memcpy(name, e->d_name, l - 7);
            name[l - 7] = 0;
            if (!good_name(name))
                continue;
            struct feed *f = feed_of(name);
            if (!f)
                continue;
            send_status(dir, f);
            send_log(dir, f);
        }
        closedir(d);
        if (fflush(stdout) != 0) {
            fprintf(stderr, "feed-relay: the pipe closed\n");
            return 1;
        }
        struct timespec ts = {1, 0};
        nanosleep(&ts, NULL);
    }
}

static int receiver(const char *dir) {
    char head[512];
    mkdir(dir, 0755);
    while (fgets(head, sizeof head, stdin)) {
        char kind, file[NAMELEN + 8];
        long long len;
        if (sscanf(head, "%c %127s %lld", &kind, file, &len) != 3 || len < 0 || len > CHUNK) {
            fprintf(stderr, "feed-relay: a frame not understood: %s", head);
            return 1;
        }
        size_t fl = strlen(file);
        int is_status = fl > 7 && !strcmp(file + fl - 7, ".status");
        int is_log = fl > 4 && !strcmp(file + fl - 4, ".log");
        if (!good_name(file) || (!is_status && !is_log) || !strcmp(file, "events.log")) {
            fprintf(stderr, "feed-relay: refused the name %s\n", file);
            return 1;
        }
        if (len && fread(buf, 1, (size_t)len, stdin) != (size_t)len) {
            fprintf(stderr, "feed-relay: the pipe closed inside a frame\n");
            return 1;
        }
        char path[4096], tmp[4200];
        snprintf(path, sizeof path, "%s/%s", dir, file);
        if (kind == 'S') {
            snprintf(tmp, sizeof tmp, "%s/.%s.relay", dir, file);
            FILE *o = fopen(tmp, "wb");
            if (!o || fwrite(buf, 1, (size_t)len, o) != (size_t)len || fclose(o) != 0 || rename(tmp, path) != 0)
                fprintf(stderr, "feed-relay: %s: %s\n", path, strerror(errno));
        } else if (kind == 'T') {
            int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0)
                close(fd);
        } else if (kind == 'A') {
            int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (fd < 0 || write(fd, buf, (size_t)len) != (ssize_t)len)
                fprintf(stderr, "feed-relay: %s: %s\n", path, strerror(errno));
            if (fd >= 0)
                close(fd);
        } else {
            fprintf(stderr, "feed-relay: an unknown frame kind %c\n", kind);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "send"))
        return sender(argv[2]);
    if (argc == 3 && !strcmp(argv[1], "recv"))
        return receiver(argv[2]);
    fprintf(stderr, "usage: feed-relay send DIR | ssh HOST feed-relay recv DIR (feed-relay.md)\n");
    return 2;
}
