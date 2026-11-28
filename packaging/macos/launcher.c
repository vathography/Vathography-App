/* Vathography App (macOS) - the bundle's main executable: starts the window with the Python inside the bundle.
 * Contents/MacOS/Vathography App -> cd Contents/Resources/app, then
 *   Contents/Resources/runtime/bin/python3.10 -B -E -s -m vathography_app gui [paths...]
 * (the same command as Windows' "Vathography App.bat"). A compiled launcher, not a shell script: its code signature
 * lives inside the file, so the zip needs no extended attributes. Built by packaging/build_package_mac.py. */
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char exe[PATH_MAX], real[PATH_MAX], contents[PATH_MAX], app[PATH_MAX], py[PATH_MAX];
    uint32_t n = sizeof exe;
    if (_NSGetExecutablePath(exe, &n) != 0 || realpath(exe, real) == NULL) {
        fprintf(stderr, "Vathography App: cannot find its own location\n");
        return 1;
    }
    char *slash = strrchr(real, '/');                 /* .../Contents/MacOS/Vathography App */
    if (slash == NULL) return 1;
    *slash = '\0';                                    /* .../Contents/MacOS */
    slash = strrchr(real, '/');
    if (slash == NULL) return 1;
    *slash = '\0';                                    /* .../Contents */
    snprintf(contents, sizeof contents, "%s", real);
    snprintf(app, sizeof app, "%s/Resources/app", contents);
    snprintf(py, sizeof py, "%s/Resources/runtime/bin/python3.10", contents);
    if (chdir(app) != 0) {
        fprintf(stderr, "Vathography App: missing %s\n", app);
        return 1;
    }
    char **args = calloc((size_t)argc + 8, sizeof(char *));
    if (args == NULL) return 1;
    int k = 0;
    args[k++] = py;
    args[k++] = "-B";                                 /* no __pycache__ written into the signed bundle (-E would */
    args[k++] = "-E";                                 /* ignore PYTHONDONTWRITEBYTECODE; the build precompiles) */
    args[k++] = "-s";
    args[k++] = "-m";
    args[k++] = "vathography_app";
    args[k++] = "gui";
    for (int i = 1; i < argc; i++)
        if (strncmp(argv[i], "-psn_", 5) != 0)        /* old Finder process-serial argument, never a path */
            args[k++] = argv[i];
    args[k] = NULL;
    execv(py, args);
    perror("Vathography App: cannot start Python");
    return 127;
}
