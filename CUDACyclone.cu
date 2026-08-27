
#include <cuda_runtime.h>
#if defined(_WIN32)
#include <device_launch_parameters.h>
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>
#include <random>
#include <algorithm>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cu"
#include "CUDAUtils.h"
#include "CUDAStructures.h"
#ifndef CUDACYCLONE_OUTLINE_RARE_VERIFY
#define CUDACYCLONE_OUTLINE_RARE_VERIFY 1
#endif
// Measured NEUTRAL on SM 12.0. A 4x180s A/B (60s ramp discarded, alternated)
// gave -1.14% in one round and +2.23% in the other: the sign flips, so this is
// noise, not a win. Kept OFF as a tuning knob only. Run-to-run spread on an
// RTX 5080 is +/-2-3% even at 180s, so do not trust deltas below ~3%.
#ifndef CUDACYCLONE_OUTLINE_HOT_HASH
#define CUDACYCLONE_OUTLINE_HOT_HASH 0
#endif

static __device__ __noinline__ uint32_t hash160_prefix_hot_outlined(
    uint8_t prefix,
    uint64_t x0,
    uint64_t x1,
    uint64_t x2,
    uint64_t x3)
{
    const uint64_t x[4] = {x0, x1, x2, x3};
    return getHash160Prefix32_33_from_limbs(prefix, x);
}

static __device__ __forceinline__ uint32_t hash160_prefix_hot(
    uint8_t prefix,
    const uint64_t x[4])
{
#if CUDACYCLONE_OUTLINE_HOT_HASH
    return hash160_prefix_hot_outlined(prefix, x[0], x[1], x[2], x[3]);
#else
    return getHash160Prefix32_33_from_limbs(prefix, x);
#endif
}

static __device__ __forceinline__ bool verify_hash160_after_prefix_match(
    uint8_t prefix,
    const uint64_t x[4])
{
#if CUDACYCLONE_OUTLINE_RARE_VERIFY
    return verifyHash160_33_from_limbs_rare(prefix, x, c_target_hash160);
#else
    uint8_t hash20[20];
    getHash160_33_from_limbs(prefix, x, hash20);
    return hash160_matches_prefix_then_full(hash20, c_target_hash160, c_target_prefix);
#endif
}

extern int cpu_avx2_worker_main(int argc, char* argv[]);

static inline bool lt256(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return false;
}

static inline void sub256_u64_host_inplace(uint64_t a[4], uint64_t dec) {
    uint64_t borrow = dec;
    for (int i = 0; i < 4 && borrow; ++i) {
        uint64_t old = a[i];
        a[i] = old - borrow;
        borrow = (old < borrow) ? 1ull : 0ull;
    }
}

static inline bool is_zero_256_host(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0ull;
}

static inline bool eq256_host(const uint64_t a[4], const uint64_t b[4]) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

static inline void copy256_host(const uint64_t src[4], uint64_t dst[4]) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
}

static inline void wrap_mod_range(uint64_t value[4], const uint64_t range_len[4]) {
    if (is_zero_256_host(range_len)) return;
    while (!lt256(value, range_len)) {
        uint64_t next[4];
        sub256(value, range_len, next);
        value[0]=next[0]; value[1]=next[1]; value[2]=next[2]; value[3]=next[3];
    }
}

static inline void random_segment_start(uint64_t out[4],
                                        const uint64_t sweep_origin[4],
                                        const uint64_t global_offset[4],
                                        const uint64_t range_start[4],
                                        const uint64_t range_len[4]) {
    uint64_t rel[4];
    sub256(sweep_origin, range_start, rel);
    add256(rel, global_offset, rel);
    wrap_mod_range(rel, range_len);
    add256(range_start, rel, out);
}

static inline void gen_random_256(uint64_t out[4], const uint64_t lo[4], const uint64_t hi[4]) {
    static thread_local std::mt19937_64 gen([]{
        std::random_device rd;
        std::seed_seq seq{
            rd(), rd(), rd(), rd(),
            (uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count(),
            (uint32_t)((uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count() >> 32)
        };
        return std::mt19937_64(seq);
    }());
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

    uint64_t len[4];
    sub256(hi, lo, len);
    add256_u64(len, 1ull, len);

    if ((len[0] | len[1] | len[2] | len[3]) == 0ull) {
        uint64_t offset[4] = { dist(gen), dist(gen), dist(gen), dist(gen) };
        add256(lo, offset, out);
        return;
    }

    int top_limb = 3;
    while (top_limb > 0 && len[top_limb] == 0ull) --top_limb;

    int highest_bit = 63;
    while (highest_bit > 0 && ((len[top_limb] >> highest_bit) & 1ull) == 0ull) --highest_bit;
    uint64_t top_mask = highest_bit == 63 ? UINT64_MAX : ((1ull << (highest_bit + 1)) - 1ull);

    uint64_t offset[4];
    do {
        for (int i = 0; i < 4; ++i) offset[i] = 0ull;
        for (int i = 0; i <= top_limb; ++i) offset[i] = dist(gen);
        offset[top_limb] &= top_mask;
    } while (!lt256(offset, len));

    add256(lo, offset, out);
}

static inline void gen_new_random_sweep_origin(uint64_t origin[4],
                                               const uint64_t range_start[4],
                                               const uint64_t range_end[4]) {
    uint64_t previous[4] = { origin[0], origin[1], origin[2], origin[3] };
    bool can_change = !eq256_host(range_start, range_end);
    for (int attempt = 0; attempt < 16; ++attempt) {
        gen_random_256(origin, range_start, range_end);
        if (!can_change || !eq256_host(origin, previous)) return;
    }
}

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int sig) { 
    g_sigint = 1; 
}

static inline bool cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(EXIT_FAILURE);
    }
    return true;
}
#define ck(e, msg) cuda_check(e, msg)

static inline uint64_t file_size_or_zero(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0ull;
    std::streampos p = f.tellg();
    return p > 0 ? (uint64_t)p : 0ull;
}

static inline std::string quote_win_arg(const std::string& s) {
#if defined(_WIN32)
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else out += c;
    }
    out += "\"";
    return out;
#else
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
#endif
}

static std::string default_self_exe_path(const char* argv0) {
#if defined(_WIN32)
    char path[4096];
    DWORD n = GetModuleFileNameA(NULL, path, (DWORD)sizeof(path));
    if (n > 0 && n < sizeof(path)) return std::string(path, path + n);
#else
    char path[4096];
    ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n > 0) {
        path[n] = '\0';
        return std::string(path);
    }
#endif
    return (argv0 != nullptr && argv0[0] != '\0') ? std::string(argv0) : std::string();
}

static std::string run_command_capture(const std::string& cmd)
{
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE read_pipe = NULL;
    HANDLE write_pipe = NULL;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) return "";
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = write_pipe;
    si.hStdError = write_pipe;

    PROCESS_INFORMATION pi{};
    std::vector<char> mutable_cmd(cmd.begin(), cmd.end());
    mutable_cmd.push_back('\0');

    BOOL ok = CreateProcessA(NULL, mutable_cmd.data(), NULL, NULL, TRUE,
                             CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
    CloseHandle(write_pipe);
    if (!ok) {
        CloseHandle(read_pipe);
        return "";
    }

    std::string out;
    char buf[4096];
    DWORD bytes_read = 0;
    while (ReadFile(read_pipe, buf, sizeof(buf), &bytes_read, NULL) && bytes_read != 0) {
        out.append(buf, buf + bytes_read);
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(read_pipe);
    return out;
#else
    std::string full_cmd = cmd + " 2>&1";
    FILE* pipe = popen(full_cmd.c_str(), "r");
    if (!pipe) return "";

    std::string out;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        out += buf;
    }
    pclose(pipe);
    return out;
#endif
}

static double parse_last_number_after_marker(const std::string& text, const std::string& marker)
{
    double last = 0.0;
    bool found = false;
    size_t pos = 0;
    while ((pos = text.find(marker, pos)) != std::string::npos) {
        const char* begin = text.c_str() + pos + marker.size();
        char* end = nullptr;
        double value = std::strtod(begin, &end);
        if (end != begin) {
            last = value;
            found = true;
        }
        pos += marker.size();
    }
    return found ? last : 0.0;
}

static std::string output_tail(const std::string& text, size_t max_chars = 2000)
{
    if (text.size() <= max_chars) return text;
    return text.substr(text.size() - max_chars);
}

static uint32_t auto_cpu_threads_all_but_one()
{
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw <= 1u) return 1u;
    return (uint32_t)(hw - 1u);
}

static std::vector<uint32_t> cpu_thread_candidates(uint32_t max_threads)
{
    std::vector<uint32_t> candidates;
    auto add = [&](uint32_t v) {
        if (v >= 1u && v <= max_threads) candidates.push_back(v);
    };

    for (uint32_t v = 1u; v <= max_threads && v <= (1u << 20); v <<= 1u) {
        add(v);
        if (v > (1u << 19)) break;
    }
    for (uint32_t v : {12u, 24u, 48u, 96u, 160u, 192u, 256u}) add(v);
    add(max_threads / 4u);
    add(max_threads / 3u);
    add(max_threads / 2u);
    add((max_threads * 2u) / 3u);
    add((max_threads * 3u) / 4u);
    add(max_threads);

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

static bool benchmark_cpu_worker(const std::string& exe_path,
                                 const std::string& address,
                                 const std::string& range_arg,
                                 uint32_t threads,
                                 uint32_t seconds,
                                 double& mkeys)
{
    std::string cmd = quote_win_arg(exe_path) +
                      " --cpu-worker" +
                      " -a " + quote_win_arg(address) +
                      " -r " + quote_win_arg(range_arg) +
                      " -t " + std::to_string(threads) +
                      " --bench-seconds " + std::to_string(seconds) +
                      " --quiet";
    std::string out = run_command_capture(cmd);
    mkeys = parse_last_number_after_marker(out, "CPU_BENCH_MKEYS=");
    if (mkeys <= 0.0) {
        std::cerr << "Error: CPU benchmark did not report a usable speed.\n"
                  << output_tail(out) << "\n";
        return false;
    }
    return true;
}

static bool tune_cpu_worker_threads(const std::string& exe_path,
                                    const std::string& address,
                                    const std::string& range_arg,
                                    uint32_t max_threads,
                                    uint32_t cpu_bench_seconds,
                                    uint32_t& best_threads)
{
    std::vector<uint32_t> candidates = cpu_thread_candidates(max_threads);
    if (candidates.empty()) {
        best_threads = max_threads > 0u ? max_threads : 1u;
        return true;
    }

    uint32_t tune_seconds = 1u;
    if (candidates.size() <= 6 && cpu_bench_seconds >= 2u) tune_seconds = 2u;

    std::cout << "Auto CPU split: tuning CPU thread count up to " << max_threads
              << " thread(s), " << tune_seconds << " second(s) per candidate...\n";

    struct Sample {
        uint32_t threads;
        double mkeys;
    };
    std::vector<Sample> samples;
    samples.reserve(candidates.size());

    double fastest_mkeys = 0.0;
    for (uint32_t t : candidates) {
        double mkeys = 0.0;
        std::cout << "  CPU tune " << t << "T...";
        std::cout.flush();
        if (!benchmark_cpu_worker(exe_path, address, range_arg, t, tune_seconds, mkeys)) {
            return false;
        }
        std::cout << " " << std::fixed << std::setprecision(1) << mkeys << " Mkeys/s\n";
        samples.push_back({t, mkeys});
        if (mkeys > fastest_mkeys) fastest_mkeys = mkeys;
    }

    const double near_best = fastest_mkeys * 0.98;
    best_threads = samples.front().threads;
    double selected_mkeys = samples.front().mkeys;
    for (const Sample& s : samples) {
        if (s.mkeys >= near_best) {
            best_threads = s.threads;
            selected_mkeys = s.mkeys;
            break;
        }
    }

    std::cout << "Auto CPU split: selected " << best_threads
              << " CPU thread(s) (" << std::fixed << std::setprecision(1)
              << selected_mkeys << " Mkeys/s tune result";
    if (best_threads != max_threads) {
        std::cout << ", lower thread count within 2% of best";
    }
    std::cout << ")\n";
    return true;
}

static bool benchmark_gpu_self(const std::string& exe_path,
                               const std::string& range_arg,
                               const std::string& gpu_list,
                               uint32_t grid_a,
                               uint32_t grid_b,
                               uint32_t slices,
                               uint32_t threads_per_block,
                               uint64_t max_launch_keys,
                               uint32_t seconds,
                               double& mkeys)
{
    static const char* dummy_hash160 = "ffffffffffffffffffffffffffffffffffffffff";
    std::string cmd = quote_win_arg(exe_path) +
                      " --range " + quote_win_arg(range_arg) +
                      " --target-hash160 " + dummy_hash160 +
                      " --grid " + std::to_string(grid_a) + "," + std::to_string(grid_b) +
                      " --slices " + std::to_string(slices) +
                      " --tpb " + std::to_string(threads_per_block) +
                      " --max-launch-keys " + std::to_string(max_launch_keys) +
                      " --seconds " + std::to_string(seconds);
    if (!gpu_list.empty()) {
        cmd += " --gpus " + quote_win_arg(gpu_list);
    }

    std::string out = run_command_capture(cmd);
    mkeys = parse_last_number_after_marker(out, "Speed:");
    if (mkeys <= 0.0) {
        std::cerr << "Error: GPU benchmark did not report a usable speed.\n"
                  << output_tail(out) << "\n";
        return false;
    }
    return true;
}

struct CpuSidecar {
    bool active = false;
    bool done = false;
    uint64_t initial_found_size = 0ull;
    std::string log_path = "cpu_worker.log";
    std::string stats_path = "cpu_worker.stats";
#if defined(_WIN32)
    PROCESS_INFORMATION pi{};
#else
    pid_t pid = -1;
#endif
};

static bool launch_cpu_sidecar(CpuSidecar& cpu,
                               const std::string& exe_path,
                               const std::string& address,
                               const std::string& range_arg,
                               uint32_t threads)
{
#if defined(_WIN32)
    cpu.initial_found_size = file_size_or_zero("found_keys.txt");
    std::remove(cpu.stats_path.c_str());
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE hLog = CreateFileA(cpu.log_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                              &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hLog == INVALID_HANDLE_VALUE) {
        std::cerr << "Error: cannot open " << cpu.log_path << " for CPU worker output.\n";
        return false;
    }

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = hLog;
    si.hStdError = hLog;

    std::string cmd = quote_win_arg(exe_path) +
                      " --cpu-worker -a " + quote_win_arg(address) +
                      " -r " + quote_win_arg(range_arg) +
                      " -t " + std::to_string(threads) +
                      " --stats-file " + quote_win_arg(cpu.stats_path) +
                      " --quiet";
    std::vector<char> mutable_cmd(cmd.begin(), cmd.end());
    mutable_cmd.push_back('\0');

    BOOL ok = CreateProcessA(NULL, mutable_cmd.data(), NULL, NULL, TRUE,
                             CREATE_NO_WINDOW, NULL, NULL, &si, &cpu.pi);
    CloseHandle(hLog);
    if (!ok) {
        std::cerr << "Error: failed to launch CPU worker: " << exe_path
                  << " (GetLastError=" << GetLastError() << ")\n";
        return false;
    }
    cpu.active = true;
    cpu.done = false;
    return true;
#else
    cpu.initial_found_size = file_size_or_zero("found_keys.txt");
    std::remove(cpu.stats_path.c_str());

    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "Error: failed to fork CPU worker: " << std::strerror(errno) << "\n";
        return false;
    }

    if (pid == 0) {
        int fd = open(cpu.log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
            close(fd);
        }

        int nullfd = open("/dev/null", O_RDONLY);
        if (nullfd >= 0) {
            dup2(nullfd, STDIN_FILENO);
            close(nullfd);
        }

        std::vector<std::string> args;
        args.push_back(exe_path);
        args.push_back("--cpu-worker");
        args.push_back("-a");
        args.push_back(address);
        args.push_back("-r");
        args.push_back(range_arg);
        args.push_back("-t");
        args.push_back(std::to_string(threads));
        args.push_back("--stats-file");
        args.push_back(cpu.stats_path);
        args.push_back("--quiet");

        std::vector<char*> argv_exec;
        argv_exec.reserve(args.size() + 1);
        for (std::string& arg : args) {
            argv_exec.push_back(arg.data());
        }
        argv_exec.push_back(nullptr);

        execvp(argv_exec[0], argv_exec.data());
        std::cerr << "Error: failed to exec CPU worker: " << exe_path
                  << " (" << std::strerror(errno) << ")\n";
        _exit(127);
    }

    cpu.pid = pid;
    cpu.active = true;
    cpu.done = false;
    return true;
#endif
}

static bool poll_cpu_sidecar(CpuSidecar& cpu, bool& found)
{
    found = false;
#if defined(_WIN32)
    if (!cpu.active) return false;
    DWORD wait = WaitForSingleObject(cpu.pi.hProcess, 0);
    if (wait == WAIT_TIMEOUT) return true;

    CloseHandle(cpu.pi.hThread);
    CloseHandle(cpu.pi.hProcess);
    cpu.active = false;
    cpu.done = true;
    found = file_size_or_zero("found_keys.txt") > cpu.initial_found_size;
    return false;
#else
    if (!cpu.active) return false;

    int status = 0;
    pid_t r = waitpid(cpu.pid, &status, WNOHANG);
    if (r == 0) return true;

    if (r == cpu.pid || (r < 0 && errno == ECHILD)) {
        cpu.active = false;
        cpu.done = true;
        found = file_size_or_zero("found_keys.txt") > cpu.initial_found_size;
        return false;
    }

    if (r < 0) {
        std::cerr << "Warning: CPU worker waitpid failed: " << std::strerror(errno) << "\n";
        cpu.active = false;
        cpu.done = true;
    }
    return false;
#endif
}

static void terminate_cpu_sidecar(CpuSidecar& cpu)
{
#if defined(_WIN32)
    if (!cpu.active) return;
    TerminateProcess(cpu.pi.hProcess, 130);
    WaitForSingleObject(cpu.pi.hProcess, 5000);
    CloseHandle(cpu.pi.hThread);
    CloseHandle(cpu.pi.hProcess);
    cpu.active = false;
    cpu.done = true;
#else
    if (!cpu.active) return;

    kill(cpu.pid, SIGTERM);
    int status = 0;
    for (int i = 0; i < 50; ++i) {
        pid_t r = waitpid(cpu.pid, &status, WNOHANG);
        if (r == cpu.pid || (r < 0 && errno == ECHILD)) {
            cpu.active = false;
            cpu.done = true;
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    kill(cpu.pid, SIGKILL);
    waitpid(cpu.pid, &status, 0);
    cpu.active = false;
    cpu.done = true;
#endif
}

static bool write_cpu_found_summary_from_file()
{
    std::ifstream in("found_keys.txt");
    if (!in) return false;

    std::string line;
    std::string last;
    while (std::getline(in, line)) {
        if (!line.empty()) last = line;
    }
    if (last.empty()) return false;

    std::istringstream iss(last);
    std::string priv, pub, wif, address;
    if (!(iss >> priv >> pub >> wif >> address)) return false;

    std::ofstream out("found_key.txt");
    if (!out) return false;
    out << "Private Key: " << priv << "\n";
    out << "Public Key: " << pub << "\n";
    out << "WIF: " << wif << "\n";
    out << "Address: " << address << "\n";
    out << "Backend: CPU AVX2\n";
    return true;
}

struct CpuLiveStats {
    bool valid = false;
    uint32_t threads = 0;
    double mkeys = 0.0;
    unsigned long long checked = 0ull;
    double elapsed = 0.0;
    double progress = 0.0;
    bool done = false;
    bool found = false;
    // Lowest key any still-running CPU thread has yet to reach. Every leftover
    // key is >= this and <= the sidecar's range end, so [remain_start, cpu_end]
    // is a superset of the sidecar's outstanding work and is safe to hand to a
    // GPU wholesale. It rescans keys the later CPU threads already cleared, but
    // the GPU is orders of magnitude faster, so that costs seconds at most.
    bool have_remaining = false;
    uint64_t remain_start[4] = {0ull, 0ull, 0ull, 0ull};
};

static bool read_cpu_live_stats(const std::string& path, CpuLiveStats& stats)
{
    std::ifstream in(path);
    if (!in) return false;

    CpuLiveStats s;
    std::string line;
    while (std::getline(in, line)) {
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        if (key == "CPU_THREADS") {
            s.threads = (uint32_t)std::strtoul(value.c_str(), nullptr, 10);
        } else if (key == "CPU_MKEYS") {
            s.mkeys = std::strtod(value.c_str(), nullptr);
        } else if (key == "CPU_CHECKED") {
            s.checked = (unsigned long long)std::strtoull(value.c_str(), nullptr, 10);
        } else if (key == "CPU_ELAPSED") {
            s.elapsed = std::strtod(value.c_str(), nullptr);
        } else if (key == "CPU_PROGRESS") {
            s.progress = std::strtod(value.c_str(), nullptr);
        } else if (key == "CPU_DONE") {
            s.done = std::strtoul(value.c_str(), nullptr, 10) != 0;
        } else if (key == "CPU_FOUND") {
            s.found = std::strtoul(value.c_str(), nullptr, 10) != 0;
        } else if (key.compare(0, 5, "CPU_T") == 0 && value != "DONE") {
            size_t colon = value.find(':');
            uint64_t begin[4];
            if (colon != std::string::npos && hexToLE64(value.substr(0, colon), begin)) {
                if (!s.have_remaining || lt256(begin, s.remain_start)) {
                    for (int k = 0; k < 4; ++k) s.remain_start[k] = begin[k];
                    s.have_remaining = true;
                }
            }
        }
    }

    s.valid = s.threads != 0 || s.mkeys > 0.0 || s.checked != 0ull;
    if (!s.valid) return false;
    stats = s;
    return true;
}

struct GPUContext {
    int deviceId;
    cudaDeviceProp prop;
    uint64_t* d_start_scalars;
    uint64_t* d_Px;
    uint64_t* d_Py;
    uint64_t* d_Rx;
    uint64_t* d_Ry;
    uint64_t* d_counts256;
    int* d_found_flag;
    FoundResult* d_found_result;
    PartialResult* d_partial_results;
    uint32_t* d_partial_count;
    uint32_t* d_partial_overflow;
    unsigned long long* d_hashes_accum;
    unsigned int* d_any_left;
    cudaEvent_t kernelDone;
    cudaStream_t stream;
    uint64_t threadsTotal;
    int blocks;
    int threadsPerBlock;
    uint64_t per_thread_cnt[4];
    uint64_t range_start[4];
    uint64_t range_len[4];
    uint64_t launchesCompleted;
    uint64_t* d_current_scalar;
    uint64_t* h_start_scalars;
    uint64_t* h_counts256;
};

static constexpr uint32_t PARTIAL_RESULT_CAPACITY = 65536u;

static inline std::string formatHash160Hex(const uint8_t hash160[20]) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 20; ++i) {
        oss << std::setw(2) << (unsigned int)hash160[i];
    }
    return oss.str();
}

static inline std::string partialOutputFile(uint32_t requested_chars, uint32_t match_chars) {
    if (match_chars <= requested_chars) return "partial.txt";
    return "partialp" + std::to_string(match_chars - requested_chars) + ".txt";
}

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
    return *((const volatile int*)p);
}
__device__ __forceinline__ bool warp_found_ready(const int* __restrict__ d_found_flag,
                                                 unsigned full_mask,
                                                 unsigned lane)
{
    int f = 0;
    if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

__device__ __forceinline__ uint8_t high_nibble(uint8_t v) { return (uint8_t)(v >> 4); }
__device__ __forceinline__ uint8_t low_nibble(uint8_t v)  { return (uint8_t)(v & 0x0Fu); }

__device__ __forceinline__ uint32_t hash160_matching_hex_chars(const uint8_t* __restrict__ h) {
    uint32_t chars = 0;
#pragma unroll
    for (int i = 0; i < 20; ++i) {
        if (high_nibble(h[i]) != high_nibble(c_target_hash160[i])) return chars;
        ++chars;
        if (low_nibble(h[i]) != low_nibble(c_target_hash160[i])) return chars;
        ++chars;
    }
    return chars;
}

__device__ __forceinline__ void record_partial_result(
    uint32_t partial_chars,
    const uint8_t* __restrict__ h20,
    const uint64_t scalar[4],
    const uint64_t X[4],
    uint8_t pubkey_prefix,
    uint64_t gid,
    PartialResult* __restrict__ partial_results,
    uint32_t* __restrict__ partial_count,
    uint32_t* __restrict__ partial_overflow,
    uint32_t partial_capacity)
{
    if (partial_chars == 0 || partial_results == nullptr || partial_count == nullptr) return;
    uint32_t matched = hash160_matching_hex_chars(h20);
    if (matched < partial_chars) return;

    uint32_t pos = atomicAdd(partial_count, 1u);
    if (pos >= partial_capacity) {
        if (partial_overflow != nullptr) atomicAdd(partial_overflow, 1u);
        return;
    }

    PartialResult* out = &partial_results[pos];
    out->match_chars = matched;
    out->threadId = (uint32_t)gid;
    out->pubkey_prefix = pubkey_prefix;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        out->scalar[k] = scalar[k];
        out->X[k] = X[k];
    }
#pragma unroll
    for (int k = 0; k < 20; ++k) out->hash160[k] = h20[k];
}

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
// 2 is optimal, measured -- do not raise it to "improve occupancy".
// sm_120, palindromic 60s runs, CUDA 13.3:
//   MIN_BLOCKS=2  REG 123  STACK 8368   512 thr/SM   5001.3 Mkeys/s
//   MIN_BLOCKS=3  REG  80  STACK 8464   768 thr/SM   4945.9  (-1.11%)
//   MIN_BLOCKS=4  REG  64  STACK 8512  1024 thr/SM   4894.9  (-2.13%)
// Cutting registers is nearly free (the frame is dominated by the 8 KB subp
// array, so 123->80 costs only +96 bytes), but the occupancy it buys is
// worthless: this kernel is ALU-throughput-bound, so extra warps add no
// throughput while multiplying local-memory traffic by the thread count.
// Note ptxas --register-usage-level is inert here; __launch_bounds__ binds first.
#ifndef HASH_FLUSH_THRESHOLD
#define HASH_FLUSH_THRESHOLD 4096u
#endif
#ifndef KERNEL_MIN_BLOCKS
#define KERNEL_MIN_BLOCKS 2
#endif
#ifndef KERNEL_MAX_THREADS
#define KERNEL_MAX_THREADS 256
#endif
#ifndef ECC_ONLY_BENCH
#define ECC_ONLY_BENCH 0
#endif
#ifndef HASH_DIAG_MODE
#define HASH_DIAG_MODE 0
#endif

__device__ __forceinline__ uint32_t hash_diag_probe(uint8_t prefix, const uint64_t x[4])
{
#if HASH_DIAG_MODE == 1
    return SHA256_33_diag_from_limbs(prefix, x);
#elif HASH_DIAG_MODE == 2
    (void)prefix;
    return RIPEMD160_diag_from_limbs(x);
#elif HASH_DIAG_MODE == 3
    // Cheapest possible consumer of the produced point. Unlike ECC_ONLY_BENCH
    // (which discards it and lets nvcc delete B-1 of the B per-batch point
    // computations), this forces every point to actually be produced, so the
    // resulting rate is the true ECC cost.
    return (uint32_t)(x[0] ^ x[1] ^ x[2] ^ x[3]) ^ (uint32_t)prefix;
#else
    (void)prefix;
    (void)x;
    return 0u;
#endif
}

__device__ unsigned int d_hash_diag_sink;

__device__ __forceinline__ void consume_hash_diag(uint32_t v)
{
    if ((v & 0x00ffffffu) == 0x00f00d00u) {
        atomicAdd(&d_hash_diag_sink, v);
    }
}

// c_Gny[i] == -c_Gy[i] mod p, i.e. fully derivable from c_Gy at ~4 ALU ops, so
// the table looks like 16 KB of __constant__ spent on nothing. Keep it anyway.
//
// Measured by static SASS census of the sm_120 hot kernel (B=512, no partial):
// dropping the table costs +16 instructions (14960 -> 14976: +10 int-ALU,
// +12 logic, +5 mov/sel) and const-load stays at 103 -- it does NOT fall.
// nvcc does not CSE the two c_Gy[i] loads across the two half-iteration blocks,
// so on-the-fly negation keeps every load it was supposed to remove and merely
// adds arithmetic on top. Strictly dominated; there is no load/ALU trade here.
//
// +0.107% of instructions is below this card's 0.13% steady-state noise floor,
// so wall-clock A/B cannot resolve it -- the census can, which is why the knob
// stays. Turning it OFF frees 16 KB of the 64 KB constant budget (Gx+Gy+Gny use
// 48 KB), which is what a MAX_BATCH_SIZE of 2048 would need.
#ifndef CUDACYCLONE_GNY_TABLE
#define CUDACYCLONE_GNY_TABLE 1
#endif
__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
#if CUDACYCLONE_GNY_TABLE
__constant__ uint64_t c_Gny[(MAX_BATCH_SIZE/2) * 4];
#endif
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];

template <int B, bool EnablePartial>
__launch_bounds__(KERNEL_MAX_THREADS, KERNEL_MIN_BLOCKS)
__global__ void kernel_point_add_and_check_oneinv(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars,
    uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
    uint32_t max_batches_per_launch,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    PartialResult* __restrict__ d_partial_results,
    uint32_t* __restrict__ d_partial_count,
    uint32_t* __restrict__ d_partial_overflow,
    uint32_t partial_chars,
    uint32_t partial_capacity,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left
)
{
    static_assert(B > 0 && (B & 1) == 0 && B <= MAX_BATCH_SIZE,
                  "B must be an even compile-time batch size within MAX_BATCH_SIZE");
    constexpr int half = B >> 1;

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    const uint32_t target_prefix = c_target_prefix;

    unsigned int local_hashes = 0;
    // A thread only pushes its tally to hashes_accum when it crosses this, or
    // at kernel end. At the old 65536 it never fired mid-launch -- a thread does
    // slices*B keys per launch, 32768 at the default 64x512 -- so the host only
    // saw the counter move at launch boundaries, roughly every 1.2 s. Sampling
    // that once a second aliased into a +-10% swing in the reported rate while
    // real throughput was steady. Must stay a power of two; the test is a mask.
    #define FLUSH_THRESHOLD HASH_FLUSH_THRESHOLD
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    uint64_t x1[4], y1[4], S[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint64_t idx = gid * 4 + i;
        x1[i] = Px[idx];
        y1[i] = Py[idx];
        S[i]  = start_scalars[idx];   
    }
    uint64_t rem[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4 + i];

    if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { Rx[gid*4+i] = x1[i]; Ry[gid*4+i] = y1[i]; }
        WARP_FLUSH_HASHES(); return;
    }

    uint32_t batches_done = 0;

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        {
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
#if ECC_ONLY_BENCH
            (void)prefix;
            ++local_hashes; MAYBE_WARP_FLUSH();
#elif HASH_DIAG_MODE != 0
            consume_hash_diag(hash_diag_probe(prefix, x1));
            ++local_hashes;
            MAYBE_WARP_FLUSH();
#else
            bool full = false;
            if constexpr (EnablePartial) {
                uint8_t h20[20];
                getHash160_33_from_limbs(prefix, x1, h20);
                record_partial_result(partial_chars, h20, S, x1, prefix, gid,
                                      d_partial_results, d_partial_count,
                                      d_partial_overflow, partial_capacity);
                full = hash160_prefix_equals(h20, target_prefix) &&
                       hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix);
            } else if (hash160_prefix_hot(prefix, x1) == target_prefix) {
                full = verify_hash160_after_prefix_match(prefix, x1);
            }
            ++local_hashes; MAYBE_WARP_FLUSH();
            if (__any_sync(full_mask, full)) {
                if (full) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=S[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=x1[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y1[k];
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }
#endif
        }

        uint64_t subp[half][4];
        uint64_t acc[4], tmp[4];

#pragma unroll
        for (int j=0;j<4;++j) acc[j] = c_Jx[j];
        ModSub256(acc, acc, x1);
#pragma unroll
        for (int j=0;j<4;++j) subp[half-1][j] = acc[j];

        for (int i = half - 2; i >= 0; --i) {
#pragma unroll
            for (int j=0;j<4;++j) tmp[j] = c_Gx[(size_t)(i+1)*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
#pragma unroll
            for (int j=0;j<4;++j) subp[i][j] = acc[j];
        }

        uint64_t d0[4], inverse[5];
#pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_Gx[0*4 + j];
        ModSub256(d0, d0, x1);
#pragma unroll
        for (int j=0;j<4;++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ull;
        _ModInv(inverse);

        uint64_t sy_neg[4], sx_neg[4];
        ModNeg256(sy_neg, y1);
        ModNeg256(sx_neg, x1);

        for (int i = 0; i < half - 1; ++i) {
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);     
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                const uint8_t prefix3 = odd ? 0x03 : 0x02;
#if ECC_ONLY_BENCH
                (void)prefix3;
                ++local_hashes; MAYBE_WARP_FLUSH();
#elif HASH_DIAG_MODE != 0
                consume_hash_diag(hash_diag_probe(prefix3, px3));
                ++local_hashes;
                MAYBE_WARP_FLUSH();
#else
                bool full = false;
                if constexpr (EnablePartial) {
                uint8_t h20[20]; getHash160_33_from_limbs(prefix3, px3, h20);
                uint64_t fs_partial[4]; for (int k=0;k<4;++k) fs_partial[k]=S[k];
                uint64_t addv_partial=(uint64_t)(i+1);
                for (int k=0;k<4 && addv_partial;++k){ uint64_t old=fs_partial[k]; fs_partial[k]=old+addv_partial; addv_partial=(fs_partial[k]<old)?1ull:0ull; }
                record_partial_result(partial_chars, h20, fs_partial, px3, prefix3, gid,
                                      d_partial_results, d_partial_count,
                                      d_partial_overflow, partial_capacity);
                full = hash160_prefix_equals(h20, target_prefix) &&
                       hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix);
            } else if (hash160_prefix_hot(prefix3, px3) == target_prefix) {
                full = verify_hash160_after_prefix_match(prefix3, px3);
                }
                ++local_hashes; MAYBE_WARP_FLUSH();
                if (__any_sync(full_mask, full)) {
                if (full) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                        uint64_t addv=(uint64_t)(i+1);
                        for (int k=0;k<4 && addv;++k){ uint64_t old=fs[k]; fs[k]=old+addv; addv=(fs[k]<old)?1ull:0ull; }
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                       
                        uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
#endif
            }

            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#if CUDACYCLONE_GNY_TABLE
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gny[(size_t)i*4+j]; }
#else
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
                ModNeg256(py_i);
#endif

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                const uint8_t prefix3 = odd ? 0x03 : 0x02;
#if ECC_ONLY_BENCH
                (void)prefix3;
                ++local_hashes; MAYBE_WARP_FLUSH();
#elif HASH_DIAG_MODE != 0
                consume_hash_diag(hash_diag_probe(prefix3, px3));
                ++local_hashes;
                MAYBE_WARP_FLUSH();
#else
                bool full = false;
                if constexpr (EnablePartial) {
                uint8_t h20[20]; getHash160_33_from_limbs(prefix3, px3, h20);
                uint64_t fs_partial[4]; for (int k=0;k<4;++k) fs_partial[k]=S[k];
                uint64_t sub_partial=(uint64_t)(i+1);
                for (int k=0;k<4 && sub_partial;++k){ uint64_t old=fs_partial[k]; fs_partial[k]=old-sub_partial; sub_partial=(old<sub_partial)?1ull:0ull; }
                record_partial_result(partial_chars, h20, fs_partial, px3, prefix3, gid,
                                      d_partial_results, d_partial_count,
                                      d_partial_overflow, partial_capacity);
                full = hash160_prefix_equals(h20, target_prefix) &&
                       hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix);
            } else if (hash160_prefix_hot(prefix3, px3) == target_prefix) {
                full = verify_hash160_after_prefix_match(prefix3, px3);
                }
                ++local_hashes; MAYBE_WARP_FLUSH();
                if (__any_sync(full_mask, full)) {
                if (full) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                        uint64_t sub=(uint64_t)(i+1);
                        for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                        uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
#endif
            }

            uint64_t gxmi[4];
#pragma unroll
            for (int j=0;j<4;++j) gxmi[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(gxmi, gxmi, x1);
            _ModMult(inverse, inverse, gxmi);
        }

        {
            const int i = half - 1;
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            uint64_t px3[4], s[4], lam[4];
            uint64_t px_i[4], py_i[4];
#if CUDACYCLONE_GNY_TABLE
#pragma unroll
            for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gny[(size_t)i*4+j]; }
#else
#pragma unroll
            for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
            ModNeg256(py_i);
#endif

            ModSub256(s, py_i, y1);
            _ModMult(lam, s, dx_inv_i);

            _ModSqr(px3, lam);
            ModSub256(px3, px3, x1);
            ModSub256(px3, px3, px_i);

            ModSub256(s, x1, px3);
            _ModMult(s, s, lam);
            uint8_t odd; ModSub256isOdd(s, y1, &odd);

            const uint8_t prefix3 = odd ? 0x03 : 0x02;
#if ECC_ONLY_BENCH
            (void)prefix3;
            ++local_hashes; MAYBE_WARP_FLUSH();
#elif HASH_DIAG_MODE != 0
            consume_hash_diag(hash_diag_probe(prefix3, px3));
            ++local_hashes;
            MAYBE_WARP_FLUSH();
#else
            bool full = false;
            if constexpr (EnablePartial) {
            uint8_t h20[20]; getHash160_33_from_limbs(prefix3, px3, h20);
            uint64_t fs_partial[4]; for (int k=0;k<4;++k) fs_partial[k]=S[k];
            uint64_t sub_partial=(uint64_t)half;
            for (int k=0;k<4 && sub_partial;++k){ uint64_t old=fs_partial[k]; fs_partial[k]=old-sub_partial; sub_partial=(old<sub_partial)?1ull:0ull; }
            record_partial_result(partial_chars, h20, fs_partial, px3, prefix3, gid,
                                  d_partial_results, d_partial_count,
                                  d_partial_overflow, partial_capacity);
            full = hash160_prefix_equals(h20, target_prefix) &&
                   hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix);
        } else if (hash160_prefix_hot(prefix3, px3) == target_prefix) {
            full = verify_hash160_after_prefix_match(prefix3, px3);
            }
            ++local_hashes; MAYBE_WARP_FLUSH();
            if (__any_sync(full_mask, full)) {
            if (full) {
                if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                    uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                    uint64_t sub=(uint64_t)half;
                    for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                    for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                    for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                    uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                    for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                    d_found_result->threadId = (int)gid;
                    d_found_result->iter     = 0;
                    __threadfence_system();
                    atomicExch(d_found_flag, FOUND_READY);
                }
            }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }
#endif

            uint64_t last_dx[4];
#pragma unroll
            for (int j=0;j<4;++j) last_dx[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(last_dx, last_dx, x1);
            _ModMult(inverse, inverse, last_dx);
        }

        {
            uint64_t lam[4], s[4], x3[4], y3[4];

            uint64_t Jy_minus_y1[4];
#pragma unroll
            for (int j=0;j<4;++j) Jy_minus_y1[j] = c_Jy[j];
            ModSub256(Jy_minus_y1, Jy_minus_y1, y1);

            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            uint64_t Jx_local[4]; for (int j=0;j<4;++j) Jx_local[j]=c_Jx[j];
            ModSub256(x3, x3, Jx_local);

            ModSub256(s, x1, x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y1);

#pragma unroll
            for (int j=0;j<4;++j) { x1[j] = x3[j]; y1[j] = y3[j]; }
        }

        {
            uint64_t addv=(uint64_t)B;
            for (int k=0;k<4 && addv;++k){ uint64_t old=S[k]; S[k]=old+addv; addv=(S[k]<old)?1ull:0ull; }
            sub256_u64_inplace(rem, (uint64_t)B);
        }
        ++batches_done;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = S[i];
    }
    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull) {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

template <int B, bool EnablePartial>
static inline void launch_point_add_and_check_oneinv_t(GPUContext& gpu,
                                                       uint32_t slices_per_launch,
                                                       uint32_t partial_digits)
{
    kernel_point_add_and_check_oneinv<B, EnablePartial><<<gpu.blocks, gpu.threadsPerBlock, 0, gpu.stream>>>(
        gpu.d_Px, gpu.d_Py, gpu.d_Rx, gpu.d_Ry,
        gpu.d_start_scalars, gpu.d_counts256,
        gpu.threadsTotal,
        slices_per_launch,
        gpu.d_found_flag, gpu.d_found_result,
        gpu.d_partial_results, gpu.d_partial_count, gpu.d_partial_overflow,
        partial_digits, PARTIAL_RESULT_CAPACITY,
        gpu.d_hashes_accum,
        gpu.d_any_left
    );
}

static inline cudaError_t launch_point_add_and_check_oneinv(GPUContext& gpu,
                                                            uint32_t batch_size,
                                                            uint32_t slices_per_launch,
                                                            uint32_t partial_digits)
{
    if (partial_digits != 0) {
        switch (batch_size) {
            case 2:    launch_point_add_and_check_oneinv_t<2, true>(gpu, slices_per_launch, partial_digits); break;
            case 4:    launch_point_add_and_check_oneinv_t<4, true>(gpu, slices_per_launch, partial_digits); break;
            case 8:    launch_point_add_and_check_oneinv_t<8, true>(gpu, slices_per_launch, partial_digits); break;
            case 16:   launch_point_add_and_check_oneinv_t<16, true>(gpu, slices_per_launch, partial_digits); break;
            case 24:   launch_point_add_and_check_oneinv_t<24, true>(gpu, slices_per_launch, partial_digits); break;
            case 32:   launch_point_add_and_check_oneinv_t<32, true>(gpu, slices_per_launch, partial_digits); break;
            case 64:   launch_point_add_and_check_oneinv_t<64, true>(gpu, slices_per_launch, partial_digits); break;
            case 128:  launch_point_add_and_check_oneinv_t<128, true>(gpu, slices_per_launch, partial_digits); break;
            case 256:  launch_point_add_and_check_oneinv_t<256, true>(gpu, slices_per_launch, partial_digits); break;
            case 512:  launch_point_add_and_check_oneinv_t<512, true>(gpu, slices_per_launch, partial_digits); break;
            case 1024: launch_point_add_and_check_oneinv_t<1024, true>(gpu, slices_per_launch, partial_digits); break;
            default:   return cudaErrorInvalidValue;
        }
    } else {
        switch (batch_size) {
            case 2:    launch_point_add_and_check_oneinv_t<2, false>(gpu, slices_per_launch, partial_digits); break;
            case 4:    launch_point_add_and_check_oneinv_t<4, false>(gpu, slices_per_launch, partial_digits); break;
            case 8:    launch_point_add_and_check_oneinv_t<8, false>(gpu, slices_per_launch, partial_digits); break;
            case 16:   launch_point_add_and_check_oneinv_t<16, false>(gpu, slices_per_launch, partial_digits); break;
            case 24:   launch_point_add_and_check_oneinv_t<24, false>(gpu, slices_per_launch, partial_digits); break;
            case 32:   launch_point_add_and_check_oneinv_t<32, false>(gpu, slices_per_launch, partial_digits); break;
            case 64:   launch_point_add_and_check_oneinv_t<64, false>(gpu, slices_per_launch, partial_digits); break;
            case 128:  launch_point_add_and_check_oneinv_t<128, false>(gpu, slices_per_launch, partial_digits); break;
            case 256:  launch_point_add_and_check_oneinv_t<256, false>(gpu, slices_per_launch, partial_digits); break;
            case 512:  launch_point_add_and_check_oneinv_t<512, false>(gpu, slices_per_launch, partial_digits); break;
            case 1024: launch_point_add_and_check_oneinv_t<1024, false>(gpu, slices_per_launch, partial_digits); break;
            default:   return cudaErrorInvalidValue;
        }
    }
    return cudaGetLastError();
}

template <bool EnablePartial>
static inline void set_point_add_kernel_cache_config_t(uint32_t batch_size)
{
    switch (batch_size) {
        case 2:    (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv<2, EnablePartial>, cudaFuncCachePreferL1); break;
        case 4:    (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv<4, EnablePartial>, cudaFuncCachePreferL1); break;
        case 8:    (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv<8, EnablePartial>, cudaFuncCachePreferL1); break;
        case 16:   (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv<16, EnablePartial>, cudaFuncCachePreferL1); break;
        case 24:   (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv<24, EnablePartial>, cudaFuncCachePreferL1); break;
        case 32:   (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv<32, EnablePartial>, cudaFuncCachePreferL1); break;
        case 64:   (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv<64, EnablePartial>, cudaFuncCachePreferL1); break;
        case 128:  (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv<128, EnablePartial>, cudaFuncCachePreferL1); break;
        case 256:  (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv<256, EnablePartial>, cudaFuncCachePreferL1); break;
        case 512:  (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv<512, EnablePartial>, cudaFuncCachePreferL1); break;
        case 1024: (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv<1024, EnablePartial>, cudaFuncCachePreferL1); break;
    }
}

static inline void set_point_add_kernel_cache_config(uint32_t batch_size, uint32_t partial_digits)
{
    if (partial_digits != 0) {
        set_point_add_kernel_cache_config_t<true>(batch_size);
    } else {
        set_point_add_kernel_cache_config_t<false>(batch_size);
    }
}

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern bool decode_p2pkh_address(const std::string& addr, uint8_t out20[20]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
extern void add256_u64_mul(const uint64_t a[4], uint64_t mult, uint64_t out[4]);
extern void divmod_256_by_u64_array(const uint64_t value[4], uint64_t divisor, uint64_t quotient[4], uint64_t remainder[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

static inline bool parse_uint32_arg(const char* text, uint32_t min_value, uint32_t max_value, uint32_t& out) {
    if (text == nullptr || *text == '\0') return false;
    uint64_t value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return false;
        value = value * 10u + (uint32_t)(*p - '0');
        if (value > max_value) return false;
    }
    if (value < min_value) return false;
    out = (uint32_t)value;
    return true;
}

static inline bool parse_uint64_arg(const char* text, uint64_t min_value, uint64_t max_value, uint64_t& out) {
    if (text == nullptr || *text == '\0') return false;
    uint64_t value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return false;
        uint64_t digit = (uint64_t)(*p - '0');
        if (value > (max_value - digit) / 10ull) return false;
        value = value * 10ull + digit;
    }
    if (value < min_value) return false;
    out = value;
    return true;
}

static inline uint64_t apply_launch_key_cap(uint64_t upper,
                                            uint64_t threads_per_block,
                                            uint32_t batch_size,
                                            uint32_t slices_per_launch,
                                            uint64_t max_launch_keys,
                                            bool& capped,
                                            uint64_t& cap_threads)
{
    capped = false;
    cap_threads = upper;
    if (max_launch_keys == 0ull || upper == 0ull) return upper;

    uint64_t keys_per_thread = (uint64_t)batch_size * (uint64_t)slices_per_launch;
    if (keys_per_thread == 0ull) return upper;

    uint64_t cap = max_launch_keys / keys_per_thread;
    if (cap < threads_per_block) cap = threads_per_block;
    cap -= cap % threads_per_block;
    if (cap < threads_per_block) cap = threads_per_block;

    cap_threads = cap;
    if (cap < upper) {
        capped = true;
        return cap;
    }
    return upper;
}

// ---------------------------------------------------------------------------
// Checkpoint / resume
//
// Thread t on a GPU starts at gpu.range_start + t*per_thread_cnt and then only
// ever walks forward in whole batches of B. The single piece of state that
// changes over time is therefore a "keys consumed per thread" counter that is
// the same for every thread, so a checkpoint is a small text header plus one
// 256-bit number per GPU -- the per-thread scalar and count arrays are rebuilt
// from it rather than dumped.
//
// Resuming is refused unless the layout that produced the file is reproduced
// exactly: same target, --range, --grid, --slices, --tpb and the same per-GPU
// thread count. Any of those re-tiles the range across threads, which would
// leave the saved offset pointing somewhere it never was.
// ---------------------------------------------------------------------------
static const char* const CHECKPOINT_MAGIC = "CUDACycloneCheckpoint";
static constexpr int CHECKPOINT_VERSION = 1;

struct CheckpointGpu {
    uint64_t threads        = 0ull;
    uint64_t start[4]       = {0ull, 0ull, 0ull, 0ull};
    uint64_t per_thread[4]  = {0ull, 0ull, 0ull, 0ull};
    uint64_t consumed[4]    = {0ull, 0ull, 0ull, 0ull};
};

struct Checkpoint {
    uint64_t range_start[4]     = {0ull, 0ull, 0ull, 0ull};
    uint64_t range_end[4]       = {0ull, 0ull, 0ull, 0ull};
    uint8_t  target_hash160[20] = {0};
    uint32_t batch_size         = 0u;
    uint32_t batches_per_sm     = 0u;
    uint32_t slices             = 0u;
    uint32_t threads_per_block  = 0u;
    uint64_t hashes             = 0ull;
    std::string cpu_tail;
    std::vector<CheckpointGpu> gpus;
};

// ---------------------------------------------------------------------------
// Checkpoint encryption
//
// A checkpoint names the address being hunted, the range, and how far the run
// has got. In the clear that is exactly what you would not want read off a
// shared box, a backup, or a recovered disk, so the body is encrypted and
// authenticated and only the file magic stays readable.
//
// The key comes from the identity of the search -- target hash160 plus range --
// which means resuming needs no extra secret: --resume already requires the same
// --address and --range on the command line. Someone holding the file without
// knowing the target cannot read it. --checkpoint-pass (or the environment
// variable CUDACYCLONE_CHECKPOINT_PASS) mixes in a passphrase as well, which
// also keeps out someone who *does* know the target.
//
// Construction: SHA-256 in counter mode for the keystream, encrypt-then-MAC with
// HMAC-SHA256, both on top of the host SHA-256 already in sha256.h. Every write
// draws a fresh random nonce -- reusing one against the same key would leak the
// XOR of two checkpoints, so the nonce must never be recycled.
// ---------------------------------------------------------------------------
// host_sha256::sha256 in sha256.h handles exactly one 64-byte block -- it was
// written for 33-byte compressed pubkeys and memcpys len bytes into a fixed
// buffer, so anything past 55 bytes runs off the end of its stack frame. The
// checkpoint KDF and HMAC hash far more than that, so they need a real
// multi-block implementation. Left as a separate function rather than fixing
// sha256.h in place, so the hot pubkey path keeps its single-block shortcut.
static void sha256_full(const uint8_t* data, size_t len, uint8_t out[32])
{
    uint32_t H[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };

    auto process = [&](const uint8_t* b) {
        uint32_t W[64];
        for (int i = 0; i < 16; ++i)
            W[i] = ((uint32_t)b[4*i+0] << 24) | ((uint32_t)b[4*i+1] << 16)
                 | ((uint32_t)b[4*i+2] <<  8) | ((uint32_t)b[4*i+3]);
        for (int t = 16; t < 64; ++t)
            W[t] = host_sha256::SSIG1(W[t-2]) + W[t-7]
                 + host_sha256::SSIG0(W[t-15]) + W[t-16];

        uint32_t a=H[0], b_=H[1], c=H[2], d=H[3], e=H[4], f=H[5], g=H[6], h=H[7];
        for (int t = 0; t < 64; ++t) {
            const uint32_t T1 = h + host_sha256::BSIG1(e) + host_sha256::Ch(e,f,g)
                              + host_sha256::K[t] + W[t];
            const uint32_t T2 = host_sha256::BSIG0(a) + host_sha256::Maj(a,b_,c);
            h=g; g=f; f=e; e=d+T1; d=c; c=b_; b_=a; a=T1+T2;
        }
        H[0]+=a; H[1]+=b_; H[2]+=c; H[3]+=d; H[4]+=e; H[5]+=f; H[6]+=g; H[7]+=h;
    };

    size_t off = 0;
    for (; off + 64 <= len; off += 64) process(data + off);

    // The 0x80 marker plus an 8-byte length need 9 bytes; if the remainder
    // leaves less than that in the block, the padding spills into a second one.
    uint8_t tail[128];
    std::memset(tail, 0, sizeof(tail));
    const size_t rem = len - off;
    if (rem) std::memcpy(tail, data + off, rem);
    tail[rem] = 0x80u;
    const size_t tail_len = (rem >= 56) ? 128u : 64u;
    const uint64_t bitlen = (uint64_t)len * 8ull;
    for (int i = 0; i < 8; ++i) tail[tail_len - 1 - i] = (uint8_t)(bitlen >> (i * 8));
    process(tail);
    if (tail_len == 128) process(tail + 64);

    for (int i = 0; i < 8; ++i) {
        out[4*i+0] = (uint8_t)(H[i] >> 24); out[4*i+1] = (uint8_t)(H[i] >> 16);
        out[4*i+2] = (uint8_t)(H[i] >>  8); out[4*i+3] = (uint8_t)(H[i]);
    }
}

static void hmac_sha256(const uint8_t* key, size_t keylen,
                        const uint8_t* msg, size_t msglen, uint8_t out[32])
{
    uint8_t k[64];
    std::memset(k, 0, sizeof(k));
    if (keylen > 64) sha256_full(key, keylen, k);
    else if (keylen)  std::memcpy(k, key, keylen);

    uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; ++i) { ipad[i] = k[i] ^ 0x36u; opad[i] = k[i] ^ 0x5cu; }

    std::vector<uint8_t> inner;
    inner.reserve(64 + msglen);
    inner.insert(inner.end(), ipad, ipad + 64);
    if (msglen) inner.insert(inner.end(), msg, msg + msglen);
    uint8_t ih[32];
    sha256_full(inner.data(), inner.size(), ih);

    uint8_t outer[64 + 32];
    std::memcpy(outer, opad, 64);
    std::memcpy(outer + 64, ih, 32);
    sha256_full(outer, sizeof(outer), out);
}

struct CheckpointKey {
    uint8_t enc[32];
    uint8_t mac[32];
};

static void derive_checkpoint_key(const uint8_t target[20],
                                  const uint64_t rs[4], const uint64_t re[4],
                                  const std::string& pass, CheckpointKey& out)
{
    static const char LABEL[] = "CUDACyclone-checkpoint-v1";
    std::vector<uint8_t> m;
    m.insert(m.end(), LABEL, LABEL + sizeof(LABEL) - 1);
    m.push_back(0x00);
    m.insert(m.end(), pass.begin(), pass.end());
    m.push_back(0x00);
    m.insert(m.end(), target, target + 20);
    for (int i = 3; i >= 0; --i) for (int b = 7; b >= 0; --b) m.push_back((uint8_t)(rs[i] >> (b * 8)));
    for (int i = 3; i >= 0; --i) for (int b = 7; b >= 0; --b) m.push_back((uint8_t)(re[i] >> (b * 8)));

    uint8_t master[32];
    sha256_full(m.data(), m.size(), master);

    uint8_t buf[33];
    std::memcpy(buf, master, 32);
    buf[32] = 0x01u; sha256_full(buf, sizeof(buf), out.enc);
    buf[32] = 0x02u; sha256_full(buf, sizeof(buf), out.mac);
}

static void checkpoint_xor_keystream(const uint8_t enc_key[32], const uint8_t nonce[16],
                                     std::vector<uint8_t>& buf)
{
    uint8_t blk[32 + 16 + 8];
    std::memcpy(blk, enc_key, 32);
    std::memcpy(blk + 32, nonce, 16);
    uint64_t ctr = 0ull;
    for (size_t off = 0; off < buf.size(); off += 32, ++ctr) {
        for (int b = 0; b < 8; ++b) blk[48 + b] = (uint8_t)(ctr >> ((7 - b) * 8));
        uint8_t ks[32];
        sha256_full(blk, sizeof(blk), ks);
        const size_t n = std::min<size_t>(32, buf.size() - off);
        for (size_t i = 0; i < n; ++i) buf[off + i] ^= ks[i];
    }
}

static std::string bytes_to_hex(const uint8_t* p, size_t n)
{
    static const char* L = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) { s.push_back(L[p[i] >> 4]); s.push_back(L[p[i] & 0xF]); }
    return s;
}

static bool hex_to_bytes(const std::string& h, std::vector<uint8_t>& out)
{
    if (h.size() % 2) return false;
    out.clear();
    out.reserve(h.size() / 2);
    for (size_t i = 0; i < h.size(); i += 2) {
        const int hi = hex_digit_value(h[i]);
        const int lo = hex_digit_value(h[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out.push_back((uint8_t)((hi << 4) | lo));
    }
    return true;
}

// Length-independent compare, so a wrong key cannot be narrowed down by timing.
static bool constant_time_equal(const uint8_t* a, const uint8_t* b, size_t n)
{
    uint8_t d = 0;
    for (size_t i = 0; i < n; ++i) d = (uint8_t)(d | (a[i] ^ b[i]));
    return d == 0;
}

static void random_nonce(uint8_t out[16])
{
    std::random_device rd;
    std::mt19937_64 gen((((uint64_t)rd() << 32) ^ (uint64_t)rd())
                        ^ (uint64_t)std::chrono::high_resolution_clock::now()
                              .time_since_epoch().count());
    for (int i = 0; i < 16; i += 8) {
        const uint64_t v = gen();
        for (int b = 0; b < 8; ++b) out[i + b] = (uint8_t)(v >> (b * 8));
    }
}

static bool write_checkpoint(const std::string& path, const Checkpoint& cp,
                             const CheckpointKey& key)
{
    std::ostringstream body;
    body << "range_start="    << formatHex256(cp.range_start)        << "\n";
    body << "range_end="      << formatHex256(cp.range_end)          << "\n";
    body << "target_hash160=" << formatHash160Hex(cp.target_hash160) << "\n";
    body << "grid="           << cp.batch_size << "," << cp.batches_per_sm << "\n";
    body << "slices="         << cp.slices                           << "\n";
    body << "tpb="            << cp.threads_per_block                << "\n";
    body << "hashes="         << cp.hashes                           << "\n";
    if (!cp.cpu_tail.empty()) body << "cpu_tail=" << cp.cpu_tail << "\n";
    body << "gpus=" << cp.gpus.size() << "\n";
    for (size_t i = 0; i < cp.gpus.size(); ++i) {
        const CheckpointGpu& g = cp.gpus[i];
        body << "gpu" << i << ".threads="    << g.threads                  << "\n";
        body << "gpu" << i << ".start="      << formatHex256(g.start)      << "\n";
        body << "gpu" << i << ".per_thread=" << formatHex256(g.per_thread) << "\n";
        body << "gpu" << i << ".consumed="   << formatHex256(g.consumed)   << "\n";
    }

    const std::string plain = body.str();
    std::vector<uint8_t> ct(plain.begin(), plain.end());

    uint8_t nonce[16];
    random_nonce(nonce);
    checkpoint_xor_keystream(key.enc, nonce, ct);

    // Encrypt-then-MAC: the tag covers the nonce as well as the ciphertext, so
    // neither can be swapped for one from another checkpoint.
    std::vector<uint8_t> signed_bytes;
    signed_bytes.reserve(16 + ct.size());
    signed_bytes.insert(signed_bytes.end(), nonce, nonce + 16);
    signed_bytes.insert(signed_bytes.end(), ct.begin(), ct.end());
    uint8_t tag[32];
    hmac_sha256(key.mac, sizeof(key.mac), signed_bytes.data(), signed_bytes.size(), tag);

    // Write to a sibling temp file and rename over the target. --autosavetimer
    // rewrites this file every N seconds while the search runs, so a crash or a
    // power cut during the write must not be able to truncate the only record of
    // hours of progress. Rename is atomic on both NTFS and POSIX.
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::trunc);
        if (!out) return false;
        out << CHECKPOINT_MAGIC << " " << CHECKPOINT_VERSION << " enc\n";
        out << "nonce=" << bytes_to_hex(nonce, 16) << "\n";
        out << "mac="   << bytes_to_hex(tag, 32)   << "\n";
        out << "data="  << bytes_to_hex(ct.data(), ct.size()) << "\n";
        out.flush();
        if (!out) { out.close(); std::remove(tmp_path.c_str()); return false; }
    }
#if defined(_WIN32)
    if (!MoveFileExA(tmp_path.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::remove(tmp_path.c_str());
        return false;
    }
#else
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        return false;
    }
#endif
    return true;
}

static bool read_checkpoint(const std::string& path, Checkpoint& cp,
                            const CheckpointKey& key, std::string& err)
{
    std::string nonce_hex, mac_hex, data_hex;
    {
        std::ifstream in(path);
        if (!in) { err = "cannot open '" + path + "'"; return false; }

        std::string line;
        if (!std::getline(in, line)) { err = "'" + path + "' is empty"; return false; }
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream hs(line);
        std::string magic, enc_tag;
        int version = 0;
        if (!(hs >> magic >> version) || magic != CHECKPOINT_MAGIC) {
            err = "'" + path + "' is not a CUDACyclone checkpoint";
            return false;
        }
        if (version != CHECKPOINT_VERSION) {
            err = "checkpoint version " + std::to_string(version) + " is not supported (expected "
                + std::to_string(CHECKPOINT_VERSION) + ")";
            return false;
        }
        if (!(hs >> enc_tag) || enc_tag != "enc") {
            err = "'" + path + "' is an old unencrypted checkpoint; delete it and start over";
            return false;
        }
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            if      (k == "nonce") nonce_hex = v;
            else if (k == "mac")   mac_hex   = v;
            else if (k == "data")  data_hex  = v;
        }
    }

    std::vector<uint8_t> nonce, tag, ct;
    if (!hex_to_bytes(nonce_hex, nonce) || nonce.size() != 16 ||
        !hex_to_bytes(mac_hex, tag)     || tag.size()   != 32 ||
        !hex_to_bytes(data_hex, ct)     || ct.empty()) {
        err = "'" + path + "' is malformed";
        return false;
    }

    std::vector<uint8_t> signed_bytes;
    signed_bytes.reserve(nonce.size() + ct.size());
    signed_bytes.insert(signed_bytes.end(), nonce.begin(), nonce.end());
    signed_bytes.insert(signed_bytes.end(), ct.begin(), ct.end());
    uint8_t expect[32];
    hmac_sha256(key.mac, sizeof(key.mac), signed_bytes.data(), signed_bytes.size(), expect);
    if (!constant_time_equal(expect, tag.data(), 32)) {
        err = "'" + path + "' will not decrypt with this run's key -- the target address or "
              "--range differs from the one it was written for (or the passphrase is wrong, "
              "or the file is damaged)";
        return false;
    }

    checkpoint_xor_keystream(key.enc, nonce.data(), ct);
    const std::string plain(ct.begin(), ct.end());

    struct KV { std::string key, value; };
    std::vector<KV> kv;
    {
        std::istringstream in(plain);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            kv.push_back(KV{ line.substr(0, eq), line.substr(eq + 1) });
        }
    }

    auto lookup = [&](const std::string& k, std::string& out)->bool {
        for (size_t i = 0; i < kv.size(); ++i) if (kv[i].key == k) { out = kv[i].value; return true; }
        return false;
    };
    auto need_hex256 = [&](const std::string& k, uint64_t out[4])->bool {
        std::string v;
        if (!lookup(k, v) || !hexToLE64(v, out)) { err = "missing or invalid '" + k + "'"; return false; }
        return true;
    };
    auto need_u32 = [&](const std::string& k, uint32_t& out)->bool {
        std::string v;
        if (!lookup(k, v) || !parse_uint32_arg(v.c_str(), 0u, UINT32_MAX, out)) {
            err = "missing or invalid '" + k + "'"; return false;
        }
        return true;
    };
    auto need_u64 = [&](const std::string& k, uint64_t& out)->bool {
        std::string v;
        if (!lookup(k, v) || !parse_uint64_arg(v.c_str(), 0ull, UINT64_MAX, out)) {
            err = "missing or invalid '" + k + "'"; return false;
        }
        return true;
    };

    if (!need_hex256("range_start", cp.range_start)) return false;
    if (!need_hex256("range_end",   cp.range_end))   return false;

    std::string v;
    if (!lookup("target_hash160", v) || !hexToHash160(v, cp.target_hash160)) {
        err = "missing or invalid 'target_hash160'"; return false;
    }
    if (!lookup("grid", v)) { err = "missing 'grid'"; return false; }
    {
        const size_t comma = v.find(',');
        if (comma == std::string::npos ||
            !parse_uint32_arg(v.substr(0, comma).c_str(), 1u, UINT32_MAX, cp.batch_size) ||
            !parse_uint32_arg(v.substr(comma + 1).c_str(), 1u, UINT32_MAX, cp.batches_per_sm)) {
            err = "invalid 'grid'"; return false;
        }
    }
    if (!need_u32("slices", cp.slices))            return false;
    if (!need_u32("tpb",    cp.threads_per_block)) return false;
    if (!need_u64("hashes", cp.hashes))            return false;
    (void)lookup("cpu_tail", cp.cpu_tail);

    uint32_t gpu_count = 0u;
    if (!need_u32("gpus", gpu_count)) return false;
    if (gpu_count == 0u || gpu_count > 64u) { err = "invalid 'gpus' count"; return false; }
    cp.gpus.resize(gpu_count);
    for (uint32_t i = 0; i < gpu_count; ++i) {
        const std::string p = "gpu" + std::to_string(i) + ".";
        if (!need_u64(p + "threads",       cp.gpus[i].threads))    return false;
        if (!need_hex256(p + "start",      cp.gpus[i].start))      return false;
        if (!need_hex256(p + "per_thread", cp.gpus[i].per_thread)) return false;
        if (!need_hex256(p + "consumed",   cp.gpus[i].consumed))   return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--cpu-worker") == 0) {
        return cpu_avx2_worker_main(argc - 1, argv + 1);
    }

    std::signal(SIGINT, handle_sigint);

    std::cout <<
R"(  __  __ _   _ _  _____ ___ ______   ______ _     ___  _   _ _____
 |  \/  | | | | ||_   _|_ _/ ___\ \ / / ___| |   / _ \| \ | | ____|
 | |\/| | | | | |  | |  | | |    \ V / |   | |  | | | |  \| |  _|
 | |  | | |_| | |__| |  | | |___  | || |___| |__| |_| | |\  | |___
 |_|  |_|\___/|____|_| |___\____| |_| \____|_____\___/|_| \_|_____|

 MULTICYCLONE v3.1 by Draikoon - forked from Dookoo2

)";

    std::string target_hash_hex, range_hex, address_b58;
    // 512 measured +7.25% over 128 on SM 12.0 (4443.6 -> 4765.5 Mkeys/s median,
    // 4x180s alternated A/B, per-round medians within 0.14%). Larger B amortizes
    // the single _ModInv over more keys; the cost is local-memory spill, which
    // bytesPerThread now accounts for so auto-sizing stays safe on small cards.
    uint32_t runtime_points_batch_size = 512;
    uint32_t runtime_batches_per_sm    = 8;
    uint32_t slices_per_launch         = 64;
    uint32_t runtime_threads_per_block = 256;
    uint32_t max_seconds               = 0;
    uint32_t autosave_seconds          = 0;
    std::string gpu_list_str;
    uint32_t random_interval_seconds   = 0;
    uint32_t partial_digits            = 0;
    bool     partial_stdout            = false;
    uint32_t cpu_threads               = 0;
    uint32_t cpu_percent               = 5;
    uint32_t cpu_bench_seconds         = 3;
    uint64_t max_launch_keys           = 6000000000ull;
    std::string self_exe_path          = default_self_exe_path(argv[0]);
    std::string cpu_exe_path           = self_exe_path;
    bool random_mode = false;
    bool resume_requested = false;
    std::string checkpoint_path = "cyclone_checkpoint.txt";
    std::string checkpoint_pass;
    if (const char* env = std::getenv("CUDACYCLONE_CHECKPOINT_PASS")) checkpoint_pass = env;
    bool cpu_auto_percent = false;
    bool cpu_threads_requested = false;
    bool cpu_threads_auto_requested = false;

    auto parse_grid = [](const std::string& s, uint32_t& a_out, uint32_t& b_out)->bool {
        size_t comma = s.find(',');
        if (comma == std::string::npos) return false;
        auto trim = [](std::string& z){
            size_t p1 = z.find_first_not_of(" \t");
            size_t p2 = z.find_last_not_of(" \t");
            if (p1 == std::string::npos) { z.clear(); return; }
            z = z.substr(p1, p2 - p1 + 1);
        };
        std::string a_str = s.substr(0, comma);
        std::string b_str = s.substr(comma + 1);
        trim(a_str); trim(b_str);
        if (a_str.empty() || b_str.empty()) return false;
        uint32_t aa = 0, bb = 0;
        if (!parse_uint32_arg(a_str.c_str(), 1u, (1u << 20), aa)) return false;
        if (!parse_uint32_arg(b_str.c_str(), 1u, (1u << 20), bb)) return false;
        a_out=aa; b_out=bb; return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      ((arg == "--target-hash160" || arg == "-target-hash160") && i + 1 < argc) target_hash_hex = argv[++i];
        else if ((arg == "--address" || arg == "-address")               && i + 1 < argc) address_b58     = argv[++i];
        else if ((arg == "--range" || arg == "-range")                   && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--resume" || arg == "-resume") resume_requested = true;
        else if ((arg == "--checkpoint" || arg == "-checkpoint") && i + 1 < argc) checkpoint_path = argv[++i];
        else if ((arg == "--checkpoint-pass" || arg == "-checkpoint-pass") && i + 1 < argc) checkpoint_pass = argv[++i];
        else if ((arg == "--grid" || arg == "-grid")                     && i + 1 < argc) {
            uint32_t a=0,b=0;
            if (!parse_grid(argv[++i], a, b)) {
                std::cerr << "Error: --grid expects \"A,B\" (positive integers).\n";
                return EXIT_FAILURE;
            }
            runtime_points_batch_size = a;
            runtime_batches_per_sm    = b;
        }
        else if ((arg == "--slices" || arg == "-slices") && i + 1 < argc) {
            uint32_t v = 0;
            if (!parse_uint32_arg(argv[++i], 1u, (1u << 20), v)) {
                std::cerr << "Error: --slices must be in 1.." << (1u<<20) << "\n";
                return EXIT_FAILURE;
            }
            slices_per_launch = v;
        }
        else if ((arg == "--tpb" || arg == "-tpb") && i + 1 < argc) {
            uint32_t v = 0;
            if (!parse_uint32_arg(argv[++i], 32u, 256u, v) || (v % 32u) != 0u) {
                std::cerr << "Error: --tpb must be one of 32,64,96,128,160,192,224,256.\n";
                return EXIT_FAILURE;
            }
            runtime_threads_per_block = v;
        }
        else if ((arg == "--seconds" || arg == "-seconds") && i + 1 < argc) {
            uint32_t v = 0;
            if (!parse_uint32_arg(argv[++i], 1u, (1u << 20), v)) {
                std::cerr << "Error: --seconds must be in 1.." << (1u<<20) << "\n";
                return EXIT_FAILURE;
            }
            max_seconds = v;
        }
        else if ((arg == "--autosavetimer" || arg == "-autosavetimer" ||
                  arg == "--autosave"      || arg == "-autosave") && i + 1 < argc) {
            uint32_t v = 0;
            if (!parse_uint32_arg(argv[++i], 1u, (1u << 20), v)) {
                std::cerr << "Error: --autosavetimer must be in 1.." << (1u<<20) << " seconds\n";
                return EXIT_FAILURE;
            }
            autosave_seconds = v;
        }
        else if ((arg == "--gpus" || arg == "-gpus") && i + 1 < argc) {
            gpu_list_str = argv[++i];
        }
        else if ((arg == "--cpu-threads" || arg == "-cpu-threads") && i + 1 < argc) {
            std::string value = argv[++i];
            if (value == "auto" || value == "all-1") {
                cpu_threads = auto_cpu_threads_all_but_one();
                cpu_threads_auto_requested = true;
                continue;
            }
            uint32_t v = 0;
            if (!parse_uint32_arg(value.c_str(), 1u, (1u << 20), v)) {
                std::cerr << "Error: --cpu-threads must be in 1.." << (1u<<20) << "\n";
                return EXIT_FAILURE;
            }
            cpu_threads = v;
            cpu_threads_requested = true;
        }
        else if ((arg == "--cpu-percent" || arg == "-cpu-percent") && i + 1 < argc) {
            std::string value = argv[++i];
            if (value == "auto") {
                cpu_auto_percent = true;
                continue;
            }
            uint32_t v = 0;
            if (!parse_uint32_arg(value.c_str(), 1u, 99u, v)) {
                std::cerr << "Error: --cpu-percent must be in 1..99.\n";
                return EXIT_FAILURE;
            }
            cpu_percent = v;
        }
        else if ((arg == "--cpu-exe" || arg == "-cpu-exe") && i + 1 < argc) {
            cpu_exe_path = argv[++i];
        }
        else if (arg == "--cpu-auto" || arg == "--auto-cpu-percent") {
            cpu_auto_percent = true;
        }
        else if ((arg == "--cpu-bench-seconds" || arg == "--bench-seconds") && i + 1 < argc) {
            uint32_t v = 0;
            if (!parse_uint32_arg(argv[++i], 1u, 3600u, v)) {
                std::cerr << "Error: --cpu-bench-seconds must be in 1..3600.\n";
                return EXIT_FAILURE;
            }
            cpu_bench_seconds = v;
        }
        else if ((arg == "--max-launch-keys" || arg == "-max-launch-keys") && i + 1 < argc) {
            uint64_t v = 0;
            if (!parse_uint64_arg(argv[++i], 0ull, UINT64_MAX, v)) {
                std::cerr << "Error: --max-launch-keys must be a non-negative integer.\n";
                return EXIT_FAILURE;
            }
            max_launch_keys = v;
        }
        else if ((arg == "--random-interval" || arg == "-random-interval") && i + 1 < argc) {
            uint32_t v = 0;
            if (!parse_uint32_arg(argv[++i], 1u, 86400u, v)) {
                std::cerr << "Error: --random-interval must be 1..86400 seconds.\n";
                return EXIT_FAILURE;
            }
            random_interval_seconds = v;
            random_mode = true;
        }
        else if ((arg == "--partial" || arg == "-partial") && i + 1 < argc) {
            uint32_t v = 0;
            if (!parse_uint32_arg(argv[++i], 1u, 40u, v)) {
                std::cerr << "Error: --partial must be a number of hash160 hex digits in 1..40.\n";
                return EXIT_FAILURE;
            }
            partial_digits = v;
        }
        else if (arg == "--partial-stdout" || arg == "-partial-stdout") {
            // Stream partial records on stdout instead of writing them to disk,
            // so candidate keys never touch the filesystem. Used by pool workers.
            partial_stdout = true;
        }
    }

    if (range_hex.empty() || (target_hash_hex.empty() && address_b58.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> (--address <base58> | --target-hash160 <hash160_hex>) [--grid A,B] [--slices N] [--tpb THREADS] [--seconds N] [--gpus GPU1,GPU2,...] [--cpu-threads N|auto] [--cpu-percent P|auto] [--cpu-auto] [--cpu-bench-seconds N] [--cpu-exe PATH] [--max-launch-keys N] [--random-interval SECONDS] [--partial HEX_DIGITS] [--resume] [--checkpoint FILE] [--checkpoint-pass PASS] [--autosavetimer SECONDS]\n";
        return EXIT_FAILURE;
    }
    if (cpu_auto_percent && !cpu_threads_requested && !cpu_threads_auto_requested) {
        cpu_threads = auto_cpu_threads_all_but_one();
        cpu_threads_auto_requested = true;
    }
    if (!target_hash_hex.empty() && !address_b58.empty()) {
        std::cerr << "Error: provide either --address or --target-hash160, not both.\n";
        return EXIT_FAILURE;
    }
    if (cpu_threads != 0 && !target_hash_hex.empty()) {
        std::cerr << "Error: --cpu-threads currently requires --address, because the AVX2 CPU backend accepts P2PKH addresses.\n";
        return EXIT_FAILURE;
    }

    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) { std::cerr << "Error: range format must be start:end\n"; return EXIT_FAILURE; }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);
    
    
    while (start_hex.length() < 64) start_hex = "0" + start_hex;
    while (end_hex.length() < 64) end_hex = "0" + end_hex;
    
    if (start_hex.length() > 64) start_hex = start_hex.substr(start_hex.length() - 64);
    if (end_hex.length() > 64) end_hex = end_hex.substr(end_hex.length() - 64);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n"; return EXIT_FAILURE;
    }
    if (lt256(range_end, range_start)) {
        std::cerr << "Error: range start must be <= range end\n";
        return EXIT_FAILURE;
    }

    std::cout << "Parsed range: " << formatHex256(range_start) << " - " << formatHex256(range_end) << std::endl;

    uint8_t target_hash160[20];
    if (!address_b58.empty()) {
        if (!decode_p2pkh_address(address_b58, target_hash160)) {
            std::cerr << "Error: invalid P2PKH address\n"; return EXIT_FAILURE;
        }
    } else {
        if (!hexToHash160(target_hash_hex, target_hash160)) {
            std::cerr << "Error: invalid target hash160 hex\n"; return EXIT_FAILURE;
        }
    }

    auto is_pow2 = [](uint32_t v)->bool { return v && ((v & (v-1)) == 0); };
    if ((runtime_points_batch_size & 1u) ||
        (!is_pow2(runtime_points_batch_size) && runtime_points_batch_size != 24u)) {
        std::cerr << "Error: batch size must be an even power of two, or the experimental 24-point group.\n";
        return EXIT_FAILURE;
    }
    if (runtime_points_batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Error: batch size must be <= " << MAX_BATCH_SIZE << " (kernel limit).\n";
        return EXIT_FAILURE;
    }

    uint64_t range_len[4]; sub256(range_end, range_start, range_len); add256_u64(range_len, 1ull, range_len);
    // The CPU sidecar split rewrites range_end/range_len below, so keep the
    // range the user actually asked for; checkpoints validate against that.
    uint64_t orig_range_end[4]; copy256_host(range_end, orig_range_end);
    uint64_t full_range_len[4];
    copy256_host(range_len, full_range_len);
    const std::string full_range_arg = formatHex256(range_start) + ":" + formatHex256(range_end);
    const std::string bench_range_arg = "1000000000000000:1FFFFFFFFFFFFFFF";

    if (cpu_auto_percent) {
        std::ifstream cpu_exe_check(cpu_exe_path, std::ios::binary);
        if (!cpu_exe_check) {
            std::cerr << "Error: CPU worker executable not found: " << cpu_exe_path << "\n";
            return EXIT_FAILURE;
        }

        if (cpu_threads_auto_requested) {
            uint32_t tuned_threads = cpu_threads;
            if (!tune_cpu_worker_threads(cpu_exe_path, address_b58, bench_range_arg,
                                         cpu_threads, cpu_bench_seconds, tuned_threads)) {
                return EXIT_FAILURE;
            }
            cpu_threads = tuned_threads;
        }

        double cpu_mkeys = 0.0;
        std::cout << "Auto CPU split: benchmarking CPU with " << cpu_threads
                  << " thread(s) for " << cpu_bench_seconds << " second(s)...\n";
        if (!benchmark_cpu_worker(cpu_exe_path, address_b58, bench_range_arg,
                                  cpu_threads, cpu_bench_seconds, cpu_mkeys)) {
            return EXIT_FAILURE;
        }

        if (random_mode) {
            std::cout << "Auto CPU random: CPU " << std::fixed << std::setprecision(1)
                      << cpu_mkeys
                      << " Mkeys/s. Random mode uses independent CPU sweeps, not a fixed range split.\n";
        } else {
            double gpu_mkeys = 0.0;
            std::cout << "Auto CPU split: benchmarking GPU with grid "
                      << runtime_points_batch_size << "," << runtime_batches_per_sm
                      << " for " << cpu_bench_seconds << " second(s)...\n";
            if (!benchmark_gpu_self(self_exe_path, bench_range_arg, gpu_list_str,
                                    runtime_points_batch_size, runtime_batches_per_sm,
                                    slices_per_launch, runtime_threads_per_block,
                                    max_launch_keys,
                                    cpu_bench_seconds, gpu_mkeys)) {
                return EXIT_FAILURE;
            }

            double share = cpu_mkeys / (cpu_mkeys + gpu_mkeys);
            uint32_t auto_percent = (uint32_t)std::floor(share * 100.0 + 0.5);
            if (auto_percent < 1u) auto_percent = 1u;
            if (auto_percent > 99u) auto_percent = 99u;
            cpu_percent = auto_percent;

            std::cout << "Auto CPU split: CPU " << std::fixed << std::setprecision(1) << cpu_mkeys
                      << " Mkeys/s, GPU " << gpu_mkeys
                      << " Mkeys/s -> CPU gets " << cpu_percent << "% of the range\n";
        }
    }

    CpuSidecar cpu_sidecar;
    bool cpu_sidecar_enabled = cpu_threads != 0;
    bool cpu_random_mode = cpu_sidecar_enabled && random_mode;
    std::string cpu_range_arg;
    uint64_t cpu_range_start[4] = {0, 0, 0, 0};
    uint64_t cpu_range_end[4] = {0, 0, 0, 0};

    if (cpu_sidecar_enabled && cpu_random_mode) {
        std::cout << "CPU sidecar: random mode keeps the full GPU range and runs independent CPU random sweeps ("
                  << cpu_threads << " thread(s)).\n";
    } else if (cpu_sidecar_enabled) {
        uint64_t hundred_q[4] = {0, 0, 0, 0};
        uint64_t hundred_rem[4] = {0, 0, 0, 0};
        divmod_256_by_u64_array(range_len, 100ull, hundred_q, hundred_rem);

        uint64_t cpu_len[4] = {0, 0, 0, 0};
        add256_u64_mul(hundred_q, (uint64_t)cpu_percent, cpu_len);
        uint64_t extra = (hundred_rem[0] * (uint64_t)cpu_percent) / 100ull;
        add256_u64(cpu_len, extra, cpu_len);
        if (is_zero_256_host(cpu_len)) {
            add256_u64(cpu_len, 1ull, cpu_len);
        }

        if (!lt256(cpu_len, range_len)) {
            copy256_host(range_len, cpu_len);
            sub256_u64_host_inplace(cpu_len, 1ull);
        }

        if (is_zero_256_host(cpu_len)) {
            std::cout << "CPU sidecar disabled: range is too small to split without overlap.\n";
            cpu_sidecar_enabled = false;
            cpu_threads = 0;
        } else {
            uint64_t cpu_len_minus_one[4];
            copy256_host(cpu_len, cpu_len_minus_one);
            sub256_u64_host_inplace(cpu_len_minus_one, 1ull);

            sub256(range_end, cpu_len_minus_one, cpu_range_start);
            copy256_host(range_end, cpu_range_end);

            uint64_t gpu_range_end[4];
            copy256_host(cpu_range_start, gpu_range_end);
            sub256_u64_host_inplace(gpu_range_end, 1ull);
            copy256_host(gpu_range_end, range_end);

            sub256(range_end, range_start, range_len);
            add256_u64(range_len, 1ull, range_len);

            cpu_range_arg = formatHex256(cpu_range_start) + ":" + formatHex256(cpu_range_end);
            std::cout << "Hybrid split: GPU range " << formatHex256(range_start)
                      << " - " << formatHex256(range_end) << "\n";
            std::cout << "Hybrid split: CPU range " << formatHex256(cpu_range_start)
                      << " - " << formatHex256(cpu_range_end)
                      << " (" << cpu_percent << "% tail, " << cpu_threads << " thread(s))\n";
        }
    }

    int device=0; cudaDeviceProp prop{};
    if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop, device)!=cudaSuccess) {
        std::cerr<<"CUDA init error\n"; return EXIT_FAILURE;
    }

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error getting CUDA device count: " << cudaGetErrorString(err) << "\n";
        return EXIT_FAILURE;
    }
    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA devices found\n";
        return EXIT_FAILURE;
    }

    std::vector<int> selectedDevices;
    if (!gpu_list_str.empty()) {
        std::stringstream ss(gpu_list_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            uint32_t parsedDev = 0;
            if (!parse_uint32_arg(item.c_str(), 0u, (uint32_t)deviceCount - 1u, parsedDev)) {
                std::cerr << "Error: Invalid device '" << item << "' (max " << deviceCount-1 << ")\n";
                return EXIT_FAILURE;
            }
            int dev = (int)parsedDev;
            if (dev < 0 || dev >= deviceCount) {
                std::cerr << "Error: Invalid device " << dev << " (max " << deviceCount-1 << ")\n";
                return EXIT_FAILURE;
            }
            selectedDevices.push_back(dev);
        }
    } else {
        for (int i = 0; i < deviceCount; ++i) selectedDevices.push_back(i);
    }

    int numGPUs = (int)selectedDevices.size();
    std::cout << "======== Multi-GPU Configuration =======================\n";
    for (int i = 0; i < numGPUs; ++i) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, selectedDevices[i]);
        std::cout << "  GPU " << i << ": " << p.name << " (compute " << p.major << "." << p.minor << ", " 
                  << p.multiProcessorCount << " SMs, " << human_bytes((double)p.totalGlobalMem) << ")\n";
    }
    std::cout << "======================================================== \n\n";

    std::vector<GPUContext> gpus;
    gpus.resize(numGPUs);

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        int devId = selectedDevices[gpuIdx];
        cudaSetDevice(devId);
        gpus[gpuIdx].deviceId = devId;
        cudaGetDeviceProperties(&gpus[gpuIdx].prop, devId);
    }

    uint64_t range_per_gpu[4];
    uint64_t range_remainder_arr[4];
    divmod_256_by_u64_array(range_len, (uint64_t)numGPUs, range_per_gpu, range_remainder_arr);
    uint64_t range_remainder = range_remainder_arr[0];

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        GPUContext& gpu = gpus[gpuIdx];
        cudaSetDevice(gpu.deviceId);

        uint64_t gpu_range_len[4];
        for (int k = 0; k < 4; ++k) gpu_range_len[k] = range_per_gpu[k];
        if ((uint64_t)gpuIdx < range_remainder) {
            add256_u64(gpu_range_len, 1ull, gpu_range_len);
        }
        for (int k = 0; k < 4; ++k) gpu.range_len[k] = gpu_range_len[k];

        uint64_t start[4];
        for (int k = 0; k < 4; ++k) start[k] = range_start[k];
        uint64_t offset[4];
        for (int k = 0; k < 4; ++k) offset[k] = range_per_gpu[k];
        add256_u64_mul(offset, (uint64_t)gpuIdx, offset);
        add256(start, offset, start);
        for (int k = 0; k < 4; ++k) gpu.range_start[k] = start[k];

        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        int threadsPerBlock = (int)runtime_threads_per_block;
        if (threadsPerBlock > (int)gpu.prop.maxThreadsPerBlock) threadsPerBlock = gpu.prop.maxThreadsPerBlock;
        if (threadsPerBlock < 32) threadsPerBlock = 32;
        threadsPerBlock -= threadsPerBlock % 32;
        if (threadsPerBlock < 32) threadsPerBlock = 32;
        gpu.threadsPerBlock = threadsPerBlock;

        // Six 4-limb arrays live per logical thread: Px, Py, Rx, Ry, counts256,
        // start_scalars. The old figure counted only two of them.
        const uint64_t bytesPerThread = 6ull*4ull*sizeof(uint64_t);
        size_t totalGlobalMem = gpu.prop.totalGlobalMem;
        const uint64_t reserveBytes = 64ull * 1024 * 1024;
        // The fused kernel spills the Montgomery prefix array subp[B/2][4] to
        // local memory. The driver backs that per *resident* thread, not per
        // logical thread, so it is a fixed reservation that must NOT scale with
        // threadsTotal -- doing so caps a 512,512 grid at 1.67M of its 11.0M
        // threads. Measured frames: B=128 -> 2448 B, 512 -> 8560, 1024 -> 16752,
        // i.e. B*16 plus a fixed ~400 B.
        const uint64_t localFrameBytes =
            (uint64_t)runtime_points_batch_size * 16ull + 512ull;
        // Residency is bounded by __launch_bounds__(KERNEL_MAX_THREADS,
        // KERNEL_MIN_BLOCKS), so only that many threads per SM can ever hold a
        // frame. Reserving for maxThreadsPerMultiProcessor instead overstates it
        // by 4x and needlessly trims the grid.
        const uint64_t residentPerSM = std::min<uint64_t>(
            (uint64_t)gpu.prop.maxThreadsPerMultiProcessor,
            (uint64_t)KERNEL_MIN_BLOCKS * (uint64_t)threadsPerBlock);
        const uint64_t localReserve =
            (uint64_t)gpu.prop.multiProcessorCount * residentPerSM * localFrameBytes;
        uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes) : (totalGlobalMem / 2);
        usableMem = (usableMem > localReserve) ? (usableMem - localReserve) : (usableMem / 2);
        uint64_t maxThreadsByMem = usableMem / bytesPerThread;

        uint64_t q_div_batch[4], r_div_batch = 0ull;
        divmod_256_by_u64(gpu_range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
        bool q_fits_u64 = (q_div_batch[3]|q_div_batch[2]|q_div_batch[1]) == 0ull;
        uint64_t total_batches_u64 = q_fits_u64 ? q_div_batch[0] : 0ull;
        if (!q_fits_u64) { 
            total_batches_u64 = UINT64_MAX;
        }

        uint64_t userUpper = (uint64_t)gpu.prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
        if (userUpper == 0ull) userUpper = UINT64_MAX;

        auto pick_threads_total = [&](uint64_t upper)->uint64_t {
            if (upper < (uint64_t)threadsPerBlock) return 0ull;
            uint64_t t = upper - (upper % (uint64_t)threadsPerBlock);
            uint64_t q = total_batches_u64;
            while (t >= (uint64_t)threadsPerBlock) {
                if (q >= t && (q % t) == 0ull) return t;
                t -= (uint64_t)threadsPerBlock;
            }
            
            t = upper - (upper % (uint64_t)threadsPerBlock);
            if (t >= (uint64_t)threadsPerBlock) return t;
            return 0ull;
        };

        uint64_t upper = maxThreadsByMem;
        if (total_batches_u64 < upper) upper = total_batches_u64;
        if (userUpper < upper) upper = userUpper;
        uint64_t upper_before_launch_cap = upper;
        bool launch_capped = false;
        uint64_t launch_cap_threads = upper;
        upper = apply_launch_key_cap(upper, (uint64_t)threadsPerBlock,
                                     runtime_points_batch_size, 1u,
                                     max_launch_keys, launch_capped, launch_cap_threads);

        uint64_t threadsTotal = pick_threads_total(upper);
        if (threadsTotal == 0ull) {
            threadsTotal = (uint64_t)threadsPerBlock;
        }
        if (launch_capped && upper_before_launch_cap > upper) {
            long double requested_keys = (long double)upper_before_launch_cap *
                                         (long double)runtime_points_batch_size;
            long double capped_keys = (long double)threadsTotal *
                                      (long double)runtime_points_batch_size;
            std::cout << "Launch guard: GPU " << gpuIdx
                      << " capped threads " << upper_before_launch_cap
                      << " -> " << threadsTotal
                      << " because one slice would exceed the launch cap (~"
                      << std::fixed << std::setprecision(1)
                      << (double)(capped_keys / 1.0e9L)
                      << "B keys/slice; requested ~"
                      << (double)(requested_keys / 1.0e9L)
                      << "B). Use --max-launch-keys 0 to disable.\n";
        }
        gpu.threadsTotal = threadsTotal;
        gpu.blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

        uint64_t per_thread_cnt[4]; uint64_t r_u64 = 0ull;
        divmod_256_by_u64(gpu_range_len, threadsTotal, per_thread_cnt, r_u64);
        if (r_u64 != 0ull) {
            add256_u64(per_thread_cnt, 1ull, per_thread_cnt);
        }
        {   uint64_t qq[4], rr=0ull;
            divmod_256_by_u64(per_thread_cnt, (uint64_t)runtime_points_batch_size, qq, rr);
            if (rr != 0ull) {
                // Round UP to the next multiple of B by adding (B - rr), not B.
                // Adding a whole batch leaves the remainder unchanged, so every
                // thread ended a launch holding 0 < rem < B. The kernel's loop
                // guard is `rem >= B`, so that tail can never be consumed, while
                // `rem != 0` keeps setting d_any_left -- the host then relaunches
                // a kernel that does no work, forever, pinned at 100%.
                add256_u64(per_thread_cnt,
                           (uint64_t)runtime_points_batch_size - rr,
                           per_thread_cnt);
            }
        }
        for (int k = 0; k < 4; ++k) gpu.per_thread_cnt[k] = per_thread_cnt[k];
    }

    const uint32_t B = runtime_points_batch_size;
    const uint32_t half = B >> 1;
    uint32_t effective_slices_per_launch = slices_per_launch;
    if (max_launch_keys != 0ull) {
        for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
            const GPUContext& gpu = gpus[gpuIdx];
            uint64_t keys_per_slice = gpu.threadsTotal * (uint64_t)B;
            uint32_t safe_slices = slices_per_launch;
            if (keys_per_slice != 0ull) {
                uint64_t q = max_launch_keys / keys_per_slice;
                if (q == 0ull) q = 1ull;
                if (q < (uint64_t)safe_slices) safe_slices = (uint32_t)q;
            }
            if (safe_slices < effective_slices_per_launch) {
                effective_slices_per_launch = safe_slices;
            }
        }
        if (effective_slices_per_launch < slices_per_launch) {
            long double keys_per_launch = 0.0L;
            for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
                keys_per_launch += (long double)gpus[gpuIdx].threadsTotal *
                                   (long double)B *
                                   (long double)effective_slices_per_launch;
            }
            std::cout << "Launch guard: using " << effective_slices_per_launch
                      << "/" << slices_per_launch
                      << " slice(s) per launch to keep kernels short (~"
                      << std::fixed << std::setprecision(1)
                      << (double)(keys_per_launch / 1.0e9L)
                      << "B keys/launch). Thread count is preserved where possible.\n";
        }
    }
    uint64_t random_sweep_origin[4] = {0, 0, 0, 0};
    uint64_t random_global_offset[4] = {0, 0, 0, 0};
    uint64_t random_sweep_coverage[4] = {0, 0, 0, 0};
    if (random_mode) {
        gen_new_random_sweep_origin(random_sweep_origin, range_start, range_end);
        std::cout << "RANDOM MODE: initial origin " << formatHex256Trimmed(random_sweep_origin) << "\n";
        for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
            uint64_t gpu_coverage[4];
            add256_u64_mul(gpus[gpuIdx].per_thread_cnt, gpus[gpuIdx].threadsTotal, gpu_coverage);
            add256(random_sweep_coverage, gpu_coverage, random_sweep_coverage);
        }
        if (!is_zero_256_host(range_len) && lt256(range_len, random_sweep_coverage) && !eq256_host(range_len, random_sweep_coverage)) {
            std::cout << "RANDOM MODE WARNING: configured sweep coverage is larger than the requested range; overlap is unavoidable.\n";
        }
    }

    // Every thread's progress is the same "keys consumed" offset, so resuming is
    // just a matter of validating that the range is tiled across threads exactly
    // as it was when the checkpoint was written, then starting each thread that
    // many keys in.
    // Keyed on the identity of the search, so --resume needs no extra secret:
    // it already demands the same --address and --range. An optional passphrase
    // is mixed in for anyone who wants the file sealed against someone who does
    // know the target.
    CheckpointKey checkpoint_key;
    derive_checkpoint_key(target_hash160, range_start, orig_range_end,
                          checkpoint_pass, checkpoint_key);

    std::vector<std::array<uint64_t,4>> resume_consumed(numGPUs);
    for (int i = 0; i < numGPUs; ++i) resume_consumed[i] = {0ull, 0ull, 0ull, 0ull};
    uint64_t resume_hashes = 0ull;

    if (resume_requested) {
        if (random_mode) {
            std::cerr << "Error: --resume is not supported with --random-interval; "
                         "random sweeps have no linear progress to resume.\n";
            return EXIT_FAILURE;
        }
        Checkpoint cp;
        std::string err;
        if (!read_checkpoint(checkpoint_path, cp, checkpoint_key, err)) {
            std::cerr << "Error: --resume failed: " << err << "\n";
            return EXIT_FAILURE;
        }

        std::vector<std::string> mismatch;
        auto cmp_u32 = [&](const char* what, uint32_t saved, uint32_t now) {
            if (saved != now) mismatch.push_back(std::string(what) + ": checkpoint "
                                                 + std::to_string(saved) + ", now " + std::to_string(now));
        };
        if (!eq256_host(cp.range_start, range_start))
            mismatch.push_back("range start: checkpoint " + formatHex256Trimmed(cp.range_start)
                               + ", now " + formatHex256Trimmed(range_start));
        if (!eq256_host(cp.range_end, orig_range_end))
            mismatch.push_back("range end: checkpoint " + formatHex256Trimmed(cp.range_end)
                               + ", now " + formatHex256Trimmed(orig_range_end));
        if (std::memcmp(cp.target_hash160, target_hash160, 20) != 0)
            mismatch.push_back("target hash160: checkpoint " + formatHash160Hex(cp.target_hash160)
                               + ", now " + formatHash160Hex(target_hash160));
        cmp_u32("grid batch size",   cp.batch_size,        runtime_points_batch_size);
        cmp_u32("grid batches/SM",   cp.batches_per_sm,    runtime_batches_per_sm);
        cmp_u32("slices",            cp.slices,            slices_per_launch);
        cmp_u32("threads per block", cp.threads_per_block, runtime_threads_per_block);
        if (cp.gpus.size() != (size_t)numGPUs) {
            mismatch.push_back("GPU count: checkpoint " + std::to_string(cp.gpus.size())
                               + ", now " + std::to_string(numGPUs));
        } else {
            for (int i = 0; i < numGPUs; ++i) {
                const std::string tag = "GPU " + std::to_string(i) + " ";
                if (cp.gpus[i].threads != gpus[i].threadsTotal)
                    mismatch.push_back(tag + "thread count: checkpoint " + std::to_string(cp.gpus[i].threads)
                                       + ", now " + std::to_string(gpus[i].threadsTotal));
                if (!eq256_host(cp.gpus[i].start, gpus[i].range_start))
                    mismatch.push_back(tag + "range start: checkpoint " + formatHex256Trimmed(cp.gpus[i].start)
                                       + ", now " + formatHex256Trimmed(gpus[i].range_start));
                if (!eq256_host(cp.gpus[i].per_thread, gpus[i].per_thread_cnt))
                    mismatch.push_back(tag + "keys/thread: checkpoint " + formatHex256Trimmed(cp.gpus[i].per_thread)
                                       + ", now " + formatHex256Trimmed(gpus[i].per_thread_cnt));
            }
        }

        if (!mismatch.empty()) {
            std::cerr << "Error: checkpoint '" << checkpoint_path << "' does not match this run:\n";
            for (size_t i = 0; i < mismatch.size(); ++i) std::cerr << "  - " << mismatch[i] << "\n";
            std::cerr << "Resume needs the same --range and target, the same --grid, --slices and\n"
                         "--tpb, and the same GPU set. Re-run with the original options, or drop\n"
                         "--resume to search the range from the beginning.\n";
            return EXIT_FAILURE;
        }

        bool all_done = true;
        for (int i = 0; i < numGPUs; ++i) {
            if (lt256(gpus[i].per_thread_cnt, cp.gpus[i].consumed)) {
                std::cerr << "Error: checkpoint '" << checkpoint_path << "' is corrupt: GPU " << i
                          << " records more keys consumed than its threads were assigned.\n";
                return EXIT_FAILURE;
            }
            for (int k = 0; k < 4; ++k) resume_consumed[i][k] = cp.gpus[i].consumed[k];
            if (!eq256_host(cp.gpus[i].consumed, gpus[i].per_thread_cnt)) all_done = false;
        }
        resume_hashes = cp.hashes;

        if (all_done) {
            std::cout << "Checkpoint '" << checkpoint_path
                      << "' is already complete; the range has been fully searched.\n";
            std::remove(checkpoint_path.c_str());
            return EXIT_SUCCESS;
        }

        long double cp_done  = ld_from_u256(cp.gpus[0].consumed);
        long double cp_total = ld_from_u256(cp.gpus[0].per_thread);
        std::cout << "Resuming from " << checkpoint_path;
        if (cp_total > 0.0L)
            std::cout << " at " << std::fixed << std::setprecision(2)
                      << (double)(cp_done / cp_total * 100.0L) << "%";
        std::cout << " (" << resume_hashes << " keys already checked)\n";
        if (!cp.cpu_tail.empty())
            std::cout << "Note: the CPU sidecar tail " << cp.cpu_tail
                      << " restarts from its beginning; only GPU progress is resumed.\n";
    }

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        GPUContext& gpu = gpus[gpuIdx];
        cudaSetDevice(gpu.deviceId);

        gpu.h_counts256 = nullptr;
        gpu.h_start_scalars = nullptr;
        cudaHostAlloc(&gpu.h_counts256,     gpu.threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        cudaHostAlloc(&gpu.h_start_scalars, gpu.threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);

        // On a fresh run resume_consumed is zero, so this is just per_thread_cnt.
        uint64_t initial_rem[4];
        sub256(gpu.per_thread_cnt, resume_consumed[gpuIdx].data(), initial_rem);
        for (uint64_t i = 0; i < gpu.threadsTotal; ++i) {
            gpu.h_counts256[i*4+0] = initial_rem[0];
            gpu.h_counts256[i*4+1] = initial_rem[1];
            gpu.h_counts256[i*4+2] = initial_rem[2];
            gpu.h_counts256[i*4+3] = initial_rem[3];
        }

        {
            uint64_t cur[4] = { gpu.range_start[0], gpu.range_start[1], gpu.range_start[2], gpu.range_start[3] };
            uint64_t cur_random_offset[4] = { random_global_offset[0], random_global_offset[1], random_global_offset[2], random_global_offset[3] };
            for (uint64_t i = 0; i < gpu.threadsTotal; ++i) {
                uint64_t base[4];
                if (random_mode) {
                    random_segment_start(base, random_sweep_origin, cur_random_offset, range_start, range_len);
                } else {
                    base[0]=cur[0]; base[1]=cur[1]; base[2]=cur[2]; base[3]=cur[3];
                }
                // base + half is where thread i starts; skip past whatever a
                // resumed checkpoint says it already covered.
                uint64_t Sc[4]; add256_u64(base, (uint64_t)half, Sc);
                add256(Sc, resume_consumed[gpuIdx].data(), Sc);
                gpu.h_start_scalars[i*4+0] = Sc[0];
                gpu.h_start_scalars[i*4+1] = Sc[1];
                gpu.h_start_scalars[i*4+2] = Sc[2];
                gpu.h_start_scalars[i*4+3] = Sc[3];

                if (!random_mode) {
                    uint64_t next[4]; add256(cur, gpu.per_thread_cnt, next);
                    cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
                } else {
                    uint64_t next[4]; add256(cur_random_offset, gpu.per_thread_cnt, next);
                    cur_random_offset[0]=next[0]; cur_random_offset[1]=next[1]; cur_random_offset[2]=next[2]; cur_random_offset[3]=next[3];
                }
            }
            if (random_mode) {
                random_global_offset[0]=cur_random_offset[0]; random_global_offset[1]=cur_random_offset[1];
                random_global_offset[2]=cur_random_offset[2]; random_global_offset[3]=cur_random_offset[3];
            }
        }

        ck(cudaMalloc(&gpu.d_start_scalars, gpu.threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_start_scalars)");
        ck(cudaMalloc(&gpu.d_Px,           gpu.threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Px)");
        ck(cudaMalloc(&gpu.d_Py,           gpu.threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Py)");
        ck(cudaMalloc(&gpu.d_Rx,           gpu.threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Rx)");
        ck(cudaMalloc(&gpu.d_Ry,           gpu.threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Ry)");
        ck(cudaMalloc(&gpu.d_counts256,    gpu.threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_counts256)");
        ck(cudaMalloc(&gpu.d_found_flag,   sizeof(int)),                         "cudaMalloc(d_found_flag)");
        ck(cudaMalloc(&gpu.d_found_result, sizeof(FoundResult)),                 "cudaMalloc(d_found_result)");
        ck(cudaMalloc(&gpu.d_partial_results, (size_t)PARTIAL_RESULT_CAPACITY * sizeof(PartialResult)), "cudaMalloc(d_partial_results)");
        ck(cudaMalloc(&gpu.d_partial_count,    sizeof(uint32_t)),                 "cudaMalloc(d_partial_count)");
        ck(cudaMalloc(&gpu.d_partial_overflow, sizeof(uint32_t)),                 "cudaMalloc(d_partial_overflow)");
        ck(cudaMalloc(&gpu.d_hashes_accum, sizeof(unsigned long long)),          "cudaMalloc(d_hashes_accum)");
        ck(cudaMalloc(&gpu.d_any_left,     sizeof(unsigned int)),                "cudaMalloc(d_any_left)");

        ck(cudaMemcpy(gpu.d_start_scalars, gpu.h_start_scalars, gpu.threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy start_scalars");
        ck(cudaMemcpy(gpu.d_counts256,     gpu.h_counts256,     gpu.threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy counts256");
        { int zero = FOUND_NONE; unsigned long long zero64=0ull; unsigned int zeroU=0u;
          ck(cudaMemcpy(gpu.d_found_flag, &zero,   sizeof(int),                cudaMemcpyHostToDevice), "init found_flag");
          ck(cudaMemcpy(gpu.d_partial_count, &zeroU, sizeof(uint32_t),          cudaMemcpyHostToDevice), "init partial_count");
          ck(cudaMemcpy(gpu.d_partial_overflow, &zeroU, sizeof(uint32_t),       cudaMemcpyHostToDevice), "init partial_overflow");
          ck(cudaMemcpy(gpu.d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init hashes_accum");
          ck(cudaMemcpy(gpu.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice), "init any_left"); }

        {
            int blocks_scal = (int)((gpu.threadsTotal + gpu.threadsPerBlock - 1) / gpu.threadsPerBlock);
            scalarMulKernelBase<<<blocks_scal, gpu.threadsPerBlock>>>(gpu.d_start_scalars, gpu.d_Px, gpu.d_Py, (int)gpu.threadsTotal);
            ck(cudaDeviceSynchronize(), "scalarMulKernelBase sync");
            ck(cudaGetLastError(), "scalarMulKernelBase launch");
        }

        {
            uint64_t* h_scalars_half = nullptr;
            cudaHostAlloc(&h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
            std::memset(h_scalars_half, 0, (size_t)half * 4 * sizeof(uint64_t));
            for (uint32_t k = 0; k < half; ++k) h_scalars_half[(size_t)k*4 + 0] = (uint64_t)(k + 1);

            uint64_t *d_scalars_half=nullptr, *d_Gx_half=nullptr, *d_Gy_half=nullptr;
            ck(cudaMalloc(&d_scalars_half, (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_scalars_half)");
            ck(cudaMalloc(&d_Gx_half,      (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gx_half)");
            ck(cudaMalloc(&d_Gy_half,      (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gy_half)");
            ck(cudaMemcpy(d_scalars_half, h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy half scalars");

            int blocks_scal = (int)((half + gpu.threadsPerBlock - 1) / gpu.threadsPerBlock);
            scalarMulKernelBase<<<blocks_scal, gpu.threadsPerBlock>>>(d_scalars_half, d_Gx_half, d_Gy_half, (int)half);
            ck(cudaDeviceSynchronize(), "scalarMulKernelBase(half) sync");
            ck(cudaGetLastError(), "scalarMulKernelBase(half) launch");

            uint64_t* h_Gx_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
            uint64_t* h_Gy_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
            ck(cudaMemcpy(h_Gx_half, d_Gx_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gx_half");
            ck(cudaMemcpy(h_Gy_half, d_Gy_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gy_half");
#if CUDACYCLONE_GNY_TABLE
            uint64_t* h_Gny_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
            const uint64_t field_p[4] = {
                0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
                0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
            };
            for (uint32_t i = 0; i < half; ++i) {
                const uint64_t* y = h_Gy_half + (size_t)i * 4;
                uint64_t* ny = h_Gny_half + (size_t)i * 4;
                if ((y[0] | y[1] | y[2] | y[3]) == 0ull) {
                    ny[0] = ny[1] = ny[2] = ny[3] = 0ull;
                } else {
                    uint64_t borrow = 0ull;
                    for (int k = 0; k < 4; ++k) {
                        const uint64_t subtrahend = y[k] + borrow;
                        const uint64_t borrow_from_add = (subtrahend < y[k]) ? 1ull : 0ull;
                        ny[k] = field_p[k] - subtrahend;
                        borrow = borrow_from_add | ((field_p[k] < subtrahend) ? 1ull : 0ull);
                    }
                }
            }
            ck(cudaMemcpyToSymbol(c_Gny, h_Gny_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gny");
            std::free(h_Gny_half);
#endif // CUDACYCLONE_GNY_TABLE
            ck(cudaMemcpyToSymbol(c_Gx, h_Gx_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gx");
            ck(cudaMemcpyToSymbol(c_Gy, h_Gy_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gy");

            cudaFree(d_scalars_half); cudaFree(d_Gx_half); cudaFree(d_Gy_half);
            cudaFreeHost(h_scalars_half);
            std::free(h_Gx_half); std::free(h_Gy_half);
        }
        {
            uint64_t* h_scalarB = nullptr;
            cudaHostAlloc(&h_scalarB, 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
            std::memset(h_scalarB, 0, 4 * sizeof(uint64_t));
            h_scalarB[0] = (uint64_t)B;

            uint64_t *d_scalarB=nullptr, *d_Jx=nullptr, *d_Jy=nullptr;
            ck(cudaMalloc(&d_scalarB, 4 * sizeof(uint64_t)), "cudaMalloc(d_scalarB)");
            ck(cudaMalloc(&d_Jx,      4 * sizeof(uint64_t)), "cudaMalloc(d_Jx)");
            ck(cudaMalloc(&d_Jy,      4 * sizeof(uint64_t)), "cudaMalloc(d_Jy)");
            ck(cudaMemcpy(d_scalarB, h_scalarB, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy scalarB");

            scalarMulKernelBase<<<1, 1>>>(d_scalarB, d_Jx, d_Jy, 1);
            ck(cudaDeviceSynchronize(), "scalarMulKernelBase(B) sync");
            ck(cudaGetLastError(), "scalarMulKernelBase(B) launch");

            uint64_t hJx[4], hJy[4];
            ck(cudaMemcpy(hJx, d_Jx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jx");
            ck(cudaMemcpy(hJy, d_Jy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jy");
            ck(cudaMemcpyToSymbol(c_Jx, hJx, 4 * sizeof(uint64_t)), "ToSymbol c_Jx");
            ck(cudaMemcpyToSymbol(c_Jy, hJy, 4 * sizeof(uint64_t)), "ToSymbol c_Jy");

            cudaFree(d_scalarB); cudaFree(d_Jx); cudaFree(d_Jy);
            cudaFreeHost(h_scalarB);
        }

        ck(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking), "create stream");
        ck(cudaEventCreateWithFlags(&gpu.kernelDone, cudaEventDisableTiming), "create event");

        uint32_t prefix_le = (uint32_t)target_hash160[0]
                           | ((uint32_t)target_hash160[1] << 8)
                           | ((uint32_t)target_hash160[2] << 16)
                           | ((uint32_t)target_hash160[3] << 24);
        cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));
        cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    }

    cudaSetDevice(gpus[0].deviceId);
    std::cout << "Single-GPU threads: " << gpus[0].threadsTotal << "\n";
    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        GPUContext& gpu = gpus[gpuIdx];
        std::cout << "GPU " << gpuIdx << ": " << gpu.prop.name << " (compute " << gpu.prop.major << "." << gpu.prop.minor << ")\n";
        std::cout << "  SMs: " << gpu.prop.multiProcessorCount << ", Threads: " << gpu.threadsTotal 
                  << ", Blocks: " << gpu.blocks << ", ThreadsPerBlock: " << gpu.threadsPerBlock << "\n";
        uint64_t gpu_end[4];
        add256(gpu.range_start, gpu.range_len, gpu_end);
        sub256_u64_host_inplace(gpu_end, 1ull);
        std::cout << "  Range: " << formatHex256(gpu.range_start) << " - " << formatHex256(gpu_end) << "\n";
    }
    std::cout << "======================================================== \n\n";
    std::cout << "======== Phase-1: BruteForce ==========================\n";
    if (random_mode) {
        std::cout << "RANDOM MODE: Re-randomizing every " << random_interval_seconds << " seconds\n\n";
    }
    if (partial_digits != 0) {
        std::cout << "Partial hash160 saving: first " << partial_digits
                  << " hex chars to partial.txt; longer matches to partialpN.txt\n\n";
    }
    set_point_add_kernel_cache_config(B, partial_digits);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;

    bool stop_all = false;
    double last_autosave_time = 0.0;
    bool   autosave_failed_warned = false;
    bool time_limit_hit = false;

    // Snapshot live GPU progress into a Checkpoint. Used both by --autosavetimer
    // during the run and by the exit path, so the two can never drift apart.
    //
    // Deliberately does NOT synchronise the stream. The host queues launches
    // ahead, so cudaStreamSynchronize here would drain the whole pipeline and
    // stall the search for seconds at every autosave. The streams are created
    // cudaStreamNonBlocking, so the plain cudaMemcpy below does not implicitly
    // sync with them either -- it can read counters a running kernel is still
    // decrementing. That is safe here because the read is only ever used
    // conservatively: see the max_rem comment below.
    auto capture_checkpoint = [&](Checkpoint& cp) -> bool {
        copy256_host(range_start, cp.range_start);
        copy256_host(orig_range_end, cp.range_end);
        std::memcpy(cp.target_hash160, target_hash160, 20);
        cp.batch_size        = runtime_points_batch_size;
        cp.batches_per_sm    = runtime_batches_per_sm;
        cp.slices            = slices_per_launch;
        cp.threads_per_block = runtime_threads_per_block;
        cp.hashes            = resume_hashes;
        cp.cpu_tail          = cpu_range_arg;
        cp.gpus.resize(numGPUs);

        for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
            GPUContext& gpu = gpus[gpuIdx];
            cudaSetDevice(gpu.deviceId);

            std::vector<uint64_t> counts((size_t)gpu.threadsTotal * 4u);
            if (cudaMemcpy(counts.data(), gpu.d_counts256,
                           gpu.threadsTotal * 4 * sizeof(uint64_t),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                return false;
            }
            // The threads march in lockstep so these should all be equal, but
            // take the largest rather than trusting that: resuming early only
            // re-scans keys, whereas resuming late would skip them. This is
            // also what makes the unsynchronised read above safe -- these
            // counters only ever decrease, so a value read mid-flight is at
            // worst stale, and the max over 11M threads is a lower bound on
            // progress. A resume can therefore repeat work but never skip it.
            uint64_t max_rem[4] = {0ull, 0ull, 0ull, 0ull};
            for (uint64_t t = 0; t < gpu.threadsTotal; ++t) {
                const uint64_t* r = &counts[(size_t)t * 4u];
                if (lt256(max_rem, r)) { for (int k = 0; k < 4; ++k) max_rem[k] = r[k]; }
            }

            unsigned long long done_hashes = 0ull;
            if (cudaMemcpy(&done_hashes, gpu.d_hashes_accum, sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                cp.hashes += done_hashes;
            }

            CheckpointGpu& g = cp.gpus[gpuIdx];
            g.threads = gpu.threadsTotal;
            copy256_host(gpu.range_start,    g.start);
            copy256_host(gpu.per_thread_cnt, g.per_thread);
            sub256(gpu.per_thread_cnt, max_rem, g.consumed);
        }
        return true;
    };
    std::atomic<bool> found_any(false);
    std::vector<unsigned long long> gpuHashes(numGPUs, 0ull);
    std::vector<bool> gpuCompleted(numGPUs, false);
    std::vector<bool> gpuNeedsLaunch(numGPUs, true);
    std::vector<uint64_t> gpuLaunches(numGPUs, 0ull);
    std::vector<uint64_t> gpuSlice(numGPUs, 0ull);
    std::vector<std::array<uint64_t,4>> gpuCurrentKey(numGPUs);
    std::vector<PartialResult> partialHost(PARTIAL_RESULT_CAPACITY);
    std::vector<unsigned long long> partialSavedByExtra(41, 0ull);
    CpuLiveStats cpuLiveStats;
    unsigned long long cpu_random_checked_base = 0ull;

    auto print_partial_live_counts = [&]() {
        if (partial_digits == 0) return;
        unsigned long long total_saved = 0ull;
        for (unsigned long long saved : partialSavedByExtra) total_saved += saved;

        std::cout << " | Partials: ";
        if (total_saved == 0ull) {
            std::cout << "0";
            return;
        }

        bool first = true;
        for (size_t extra = 0; extra < partialSavedByExtra.size(); ++extra) {
            unsigned long long saved = partialSavedByExtra[extra];
            if (saved == 0ull) continue;
            if (!first) std::cout << " ";
            if (extra == 0) {
                std::cout << "partial.txt=" << saved;
            } else {
                std::cout << "partialp" << extra << ".txt=" << saved;
            }
            first = false;
        }
    };

    auto drain_partial_results = [&](GPUContext& gpu, int gpuIdx) {
        if (partial_digits == 0) return;

        uint32_t count = 0;
        uint32_t overflow = 0;
        ck(cudaMemcpy(&count, gpu.d_partial_count, sizeof(uint32_t), cudaMemcpyDeviceToHost), "read partial_count");
        ck(cudaMemcpy(&overflow, gpu.d_partial_overflow, sizeof(uint32_t), cudaMemcpyDeviceToHost), "read partial_overflow");

        uint32_t to_copy = std::min(count, PARTIAL_RESULT_CAPACITY);
        if (to_copy != 0) {
            ck(cudaMemcpy(partialHost.data(), gpu.d_partial_results, (size_t)to_copy * sizeof(PartialResult), cudaMemcpyDeviceToHost), "read partial_results");
        }

        uint32_t zeroU = 0u;
        ck(cudaMemcpy(gpu.d_partial_count, &zeroU, sizeof(uint32_t), cudaMemcpyHostToDevice), "reset partial_count");
        ck(cudaMemcpy(gpu.d_partial_overflow, &zeroU, sizeof(uint32_t), cudaMemcpyHostToDevice), "reset partial_overflow");

        for (uint32_t i = 0; i < to_copy; ++i) {
            const PartialResult& r = partialHost[i];
            uint32_t extra = (r.match_chars > partial_digits) ? (r.match_chars - partial_digits) : 0u;
            if (extra >= partialSavedByExtra.size()) extra = (uint32_t)partialSavedByExtra.size() - 1u;
            ++partialSavedByExtra[extra];
            if (partial_stdout) {
                // The progress line ends in a carriage return, so lead with a
                // newline to keep each record on a line of its own.
                std::cout << "\nPARTIAL " << formatHex256(r.scalar)
                          << " " << formatHash160Hex(r.hash160) << "\n";
                std::cout.flush();
            } else {
                std::ofstream out(partialOutputFile(partial_digits, r.match_chars), std::ios::app);
                out << "Hex: " << formatHex256(r.scalar)
                    << " Hash160: " << formatHash160Hex(r.hash160) << "\n";
            }
        }

        if (overflow != 0) {
            std::cout << "\nPartial result buffer overflow on GPU " << gpuIdx
                      << ": " << overflow << " candidate(s) were not saved. Increase --partial to reduce hit rate.\n";
        }
    };

    auto add_cpu_random_checked_to_base = [&]() {
        if (!cpu_random_mode) return;
        CpuLiveStats stats;
        if (read_cpu_live_stats(cpu_sidecar.stats_path, stats)) {
            cpu_random_checked_base += stats.checked;
        }
    };

    auto make_random_cpu_range_arg = [&]()->std::string {
        uint64_t cpu_origin[4] = {0, 0, 0, 0};
        gen_new_random_sweep_origin(cpu_origin, range_start, range_end);
        return formatHex256(cpu_origin) + ":" + formatHex256(range_end);
    };

    auto launch_cpu_sidecar_current = [&](bool announce)->bool {
        std::string launch_range = cpu_range_arg;
        if (cpu_random_mode) {
            launch_range = make_random_cpu_range_arg();
            cpu_range_arg = launch_range;
            if (announce) {
                std::cout << "CPU sidecar random range: " << launch_range << "\n";
            }
        }
        return launch_cpu_sidecar(cpu_sidecar, cpu_exe_path, address_b58, launch_range, cpu_threads);
    };

    auto restart_cpu_random_sidecar = [&]()->bool {
        if (!cpu_random_mode || !cpu_sidecar_enabled) return true;
        if (cpu_sidecar.active) {
            add_cpu_random_checked_to_base();
            terminate_cpu_sidecar(cpu_sidecar);
        }
        return launch_cpu_sidecar_current(true);
    };

    double last_random_time = 0.0;
    auto rerandomize_all_gpus = [&]()->bool {
        gen_new_random_sweep_origin(random_sweep_origin, range_start, range_end);
        std::cout << "\n[RANDOM MODE] Re-randomizing keys: origin "
                  << formatHex256Trimmed(random_sweep_origin) << "\n";

        uint64_t global_offset[4] = {0, 0, 0, 0};
        for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
            GPUContext& gpu = gpus[gpuIdx];
            cudaSetDevice(gpu.deviceId);
            cudaStreamSynchronize(gpu.stream);
            drain_partial_results(gpu, gpuIdx);

            uint64_t cur_offset[4] = { global_offset[0], global_offset[1], global_offset[2], global_offset[3] };
            for (uint64_t t = 0; t < gpu.threadsTotal; ++t) {
                uint64_t base_key[4];
                random_segment_start(base_key, random_sweep_origin, cur_offset, range_start, range_len);

                uint64_t Sc[4];
                add256_u64(base_key, half, Sc);
                gpu.h_start_scalars[t*4+0] = Sc[0];
                gpu.h_start_scalars[t*4+1] = Sc[1];
                gpu.h_start_scalars[t*4+2] = Sc[2];
                gpu.h_start_scalars[t*4+3] = Sc[3];

                uint64_t next_offset[4];
                add256(cur_offset, gpu.per_thread_cnt, next_offset);
                cur_offset[0]=next_offset[0]; cur_offset[1]=next_offset[1];
                cur_offset[2]=next_offset[2]; cur_offset[3]=next_offset[3];
            }
            global_offset[0]=cur_offset[0]; global_offset[1]=cur_offset[1];
            global_offset[2]=cur_offset[2]; global_offset[3]=cur_offset[3];

            uint64_t sampleIdx = gpu.threadsTotal > 1 ? (gpu.threadsTotal / 2) : 0;
            gpuCurrentKey[gpuIdx][0] = gpu.h_start_scalars[sampleIdx*4+0];
            gpuCurrentKey[gpuIdx][1] = gpu.h_start_scalars[sampleIdx*4+1];
            gpuCurrentKey[gpuIdx][2] = gpu.h_start_scalars[sampleIdx*4+2];
            gpuCurrentKey[gpuIdx][3] = gpu.h_start_scalars[sampleIdx*4+3];

            unsigned int zeroU = 0u;
            int zero = FOUND_NONE;
            ck(cudaMemcpy(gpu.d_start_scalars, gpu.h_start_scalars,
                          gpu.threadsTotal * 4 * sizeof(uint64_t),
                          cudaMemcpyHostToDevice), "random cpy start_scalars");
            ck(cudaMemcpy(gpu.d_counts256, gpu.h_counts256,
                          gpu.threadsTotal * 4 * sizeof(uint64_t),
                          cudaMemcpyHostToDevice), "random cpy counts256");
            ck(cudaMemcpy(gpu.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice), "random reset any_left");
            ck(cudaMemcpy(gpu.d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice), "random reset found_flag");

            int blocks_scal = (int)((gpu.threadsTotal + gpu.threadsPerBlock - 1) / gpu.threadsPerBlock);
            scalarMulKernelBase<<<blocks_scal, gpu.threadsPerBlock>>>(gpu.d_start_scalars, gpu.d_Px, gpu.d_Py, (int)gpu.threadsTotal);
            ck(cudaDeviceSynchronize(), "random scalarMulKernelBase sync");
            ck(cudaGetLastError(), "random scalarMulKernelBase launch");

            cudaMemcpyAsync(gpu.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, gpu.stream);
            cudaError_t launchErr = launch_point_add_and_check_oneinv(gpu, B, effective_slices_per_launch, partial_digits);
            if (launchErr != cudaSuccess) {
                std::cerr << "\nKernel launch error on GPU " << gpuIdx << ": "
                          << cudaGetErrorString(launchErr) << "\n";
                return false;
            }
            cudaEventRecord(gpu.kernelDone, gpu.stream);
            gpuSlice[gpuIdx] = 0;
            gpuCompleted[gpuIdx] = false;
            gpuNeedsLaunch[gpuIdx] = false;
        }
        if (!restart_cpu_random_sidecar()) {
            std::cerr << "Error: failed to restart CPU sidecar for random sweep.\n";
            return false;
        }
        last_random_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        std::cout.flush();
        return true;
    };

    // Point every GPU at a fresh sub-range and restart them on it. Thread counts
    // and every device allocation stay as they are -- only the per-thread start
    // scalars and remaining counts are rebuilt -- so this is cheap enough to use
    // for taking over the CPU sidecar's leftovers.
    auto retarget_gpus_to_range = [&](const uint64_t new_start[4], const uint64_t new_len[4])->bool {
        uint64_t per_gpu[4], rem_arr[4];
        divmod_256_by_u64_array(new_len, (uint64_t)numGPUs, per_gpu, rem_arr);
        const uint64_t split_rem = rem_arr[0];

        uint64_t cursor[4];
        copy256_host(new_start, cursor);

        for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
            GPUContext& gpu = gpus[gpuIdx];
            cudaSetDevice(gpu.deviceId);
            cudaStreamSynchronize(gpu.stream);
            drain_partial_results(gpu, gpuIdx);

            uint64_t gpu_len[4];
            copy256_host(per_gpu, gpu_len);
            if ((uint64_t)gpuIdx < split_rem) add256_u64(gpu_len, 1ull, gpu_len);

            copy256_host(cursor, gpu.range_start);
            copy256_host(gpu_len, gpu.range_len);
            add256(cursor, gpu_len, cursor);

            // Same rounding rule as the initial setup: whole batches of B only,
            // or the kernel's `rem >= B` guard could never drain the tail.
            uint64_t ptc[4]; uint64_t r64 = 0ull;
            divmod_256_by_u64(gpu_len, gpu.threadsTotal, ptc, r64);
            if (r64 != 0ull) add256_u64(ptc, 1ull, ptc);
            {   uint64_t qq[4], rr = 0ull;
                divmod_256_by_u64(ptc, (uint64_t)B, qq, rr);
                if (rr != 0ull) add256_u64(ptc, (uint64_t)B - rr, ptc);
            }
            if (is_zero_256_host(ptc)) add256_u64(ptc, (uint64_t)B, ptc);
            copy256_host(ptc, gpu.per_thread_cnt);

            uint64_t cur[4];
            copy256_host(gpu.range_start, cur);
            for (uint64_t t = 0; t < gpu.threadsTotal; ++t) {
                uint64_t Sc[4];
                add256_u64(cur, (uint64_t)half, Sc);
                gpu.h_start_scalars[t*4+0] = Sc[0];
                gpu.h_start_scalars[t*4+1] = Sc[1];
                gpu.h_start_scalars[t*4+2] = Sc[2];
                gpu.h_start_scalars[t*4+3] = Sc[3];
                gpu.h_counts256[t*4+0] = ptc[0];
                gpu.h_counts256[t*4+1] = ptc[1];
                gpu.h_counts256[t*4+2] = ptc[2];
                gpu.h_counts256[t*4+3] = ptc[3];
                uint64_t next[4]; add256(cur, ptc, next);
                copy256_host(next, cur);
            }

            uint64_t sampleIdx = gpu.threadsTotal > 1 ? (gpu.threadsTotal / 2) : 0;
            gpuCurrentKey[gpuIdx][0] = gpu.h_start_scalars[sampleIdx*4+0];
            gpuCurrentKey[gpuIdx][1] = gpu.h_start_scalars[sampleIdx*4+1];
            gpuCurrentKey[gpuIdx][2] = gpu.h_start_scalars[sampleIdx*4+2];
            gpuCurrentKey[gpuIdx][3] = gpu.h_start_scalars[sampleIdx*4+3];

            unsigned int zeroU = 0u;
            int zero = FOUND_NONE;
            ck(cudaMemcpy(gpu.d_start_scalars, gpu.h_start_scalars,
                          gpu.threadsTotal * 4 * sizeof(uint64_t),
                          cudaMemcpyHostToDevice), "retarget cpy start_scalars");
            ck(cudaMemcpy(gpu.d_counts256, gpu.h_counts256,
                          gpu.threadsTotal * 4 * sizeof(uint64_t),
                          cudaMemcpyHostToDevice), "retarget cpy counts256");
            ck(cudaMemcpy(gpu.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice), "retarget reset any_left");
            ck(cudaMemcpy(gpu.d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice), "retarget reset found_flag");

            int blocks_scal = (int)((gpu.threadsTotal + gpu.threadsPerBlock - 1) / gpu.threadsPerBlock);
            scalarMulKernelBase<<<blocks_scal, gpu.threadsPerBlock>>>(gpu.d_start_scalars, gpu.d_Px, gpu.d_Py, (int)gpu.threadsTotal);
            ck(cudaDeviceSynchronize(), "retarget scalarMulKernelBase sync");
            ck(cudaGetLastError(), "retarget scalarMulKernelBase launch");

            cudaMemcpyAsync(gpu.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, gpu.stream);
            cudaError_t launchErr = launch_point_add_and_check_oneinv(gpu, B, effective_slices_per_launch, partial_digits);
            if (launchErr != cudaSuccess) {
                std::cerr << "\nKernel launch error on GPU " << gpuIdx << ": "
                          << cudaGetErrorString(launchErr) << "\n";
                return false;
            }
            cudaEventRecord(gpu.kernelDone, gpu.stream);
            gpuSlice[gpuIdx] = 0;
            gpuCompleted[gpuIdx] = false;
            gpuNeedsLaunch[gpuIdx] = false;
        }
        return true;
    };

    // Fired when every GPU has finished its own share while the CPU sidecar is
    // still grinding through its tail. Rather than idle, stop the sidecar and let
    // the GPUs sweep everything it had left.
    bool cpu_work_taken_over = false;
    auto take_over_cpu_remainder = [&]()->bool {
        CpuLiveStats leftover;
        bool have = read_cpu_live_stats(cpu_sidecar.stats_path, leftover);
        terminate_cpu_sidecar(cpu_sidecar);
        cpu_sidecar.active = false;
        cpu_work_taken_over = true;

        if (have && leftover.checked != 0ull) cpu_random_checked_base += leftover.checked;

        uint64_t from[4];
        if (have && leftover.have_remaining) {
            copy256_host(leftover.remain_start, from);
        } else {
            // No usable per-thread report: fall back to the whole sidecar range.
            copy256_host(cpu_range_start, from);
        }
        if (lt256(from, cpu_range_start)) copy256_host(cpu_range_start, from);
        if (lt256(cpu_range_end, from)) {
            std::cout << "\nGPUs idle; CPU sidecar had already finished its tail.\n";
            return true;
        }

        uint64_t takeover_len[4];
        sub256(cpu_range_end, from, takeover_len);
        add256_u64(takeover_len, 1ull, takeover_len);

        std::cout << "\nGPUs finished their share; taking over the CPU sidecar's remaining "
                  << formatHex256Trimmed(from) << " - " << formatHex256Trimmed(cpu_range_end)
                  << " (" << std::fixed << std::setprecision(2)
                  << (double)(ld_from_u256(takeover_len) / 1.0e9L) << "B keys)\n";
        std::cout.flush();
        return retarget_gpus_to_range(from, takeover_len);
    };

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        GPUContext& gpu = gpus[gpuIdx];
        cudaSetDevice(gpu.deviceId);
        
        unsigned int zeroU = 0u;
        ck(cudaMemcpyAsync(gpu.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, gpu.stream), "zero d_any_left");
        
        cudaError_t launchErr = launch_point_add_and_check_oneinv(gpu, B, effective_slices_per_launch, partial_digits);
        if (launchErr != cudaSuccess) {
            std::cerr << "\nKernel launch error on GPU " << gpuIdx << ": " << cudaGetErrorString(launchErr) << "\n";
            return EXIT_FAILURE;
        }
        cudaEventRecord(gpu.kernelDone, gpu.stream);
        gpuNeedsLaunch[gpuIdx] = false;
    }

    if (cpu_sidecar_enabled) {
        std::ifstream cpu_exe_check(cpu_exe_path, std::ios::binary);
        if (!cpu_exe_check) {
            std::cerr << "Error: CPU worker executable not found: " << cpu_exe_path << "\n";
            return EXIT_FAILURE;
        }
        if (cpu_random_mode) {
            std::cout << "CPU sidecar: launching " << cpu_threads
                      << " AVX2 thread(s) for random sweeps\n";
        } else {
            std::cout << "CPU sidecar: launching " << cpu_threads << " AVX2 thread(s), "
                      << cpu_percent << "% tail range\n";
        }
        std::cout << "CPU sidecar log: cpu_worker.log\n";
        if (!launch_cpu_sidecar_current(true)) {
            return EXIT_FAILURE;
        }
    }

    while (!stop_all) {
        if (g_sigint) {
            std::cerr << "\n[Ctrl+C] Interrupt received. Finishing current kernel slices and exiting...\n";
            stop_all = true;
        }

        if (cpu_sidecar.active) {
            bool cpu_found = false;
            bool cpu_running = poll_cpu_sidecar(cpu_sidecar, cpu_found);
            if (!cpu_running) {
                if (cpu_found) {
                    stop_all = true;
                    found_any.store(true);
                    std::cout << "\n======== FOUND MATCH! =================================\n";
                    std::cout << "Found on CPU AVX2 sidecar\n";
                    if (write_cpu_found_summary_from_file()) {
                        std::cout << "Result saved to found_key.txt\n";
                    }
                    std::cout << "Full CPU output: cpu_worker.log\n";
                } else if (cpu_random_mode && !stop_all) {
                    add_cpu_random_checked_to_base();
                    std::cout << "\nCPU sidecar completed random sweep; starting a new one.\n";
                    if (!launch_cpu_sidecar_current(true)) {
                        stop_all = true;
                    }
                } else if (!stop_all) {
                    std::cout << "\nCPU sidecar completed its assigned range.\n";
                }
            }
        }

        if (stop_all) break;

        bool any_stream_busy = false;
        unsigned long long totalHashes = 0ull;

        for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
            if (gpuCompleted[gpuIdx]) {
                totalHashes += gpuHashes[gpuIdx];
                continue;
            }
            GPUContext& gpu = gpus[gpuIdx];
            cudaSetDevice(gpu.deviceId);

            cudaError_t qs = cudaEventQuery(gpu.kernelDone);
            if (qs == cudaSuccess) {
                cudaDeviceSynchronize();
                drain_partial_results(gpu, gpuIdx);
                unsigned int h_any = 0u;
                ck(cudaMemcpy(&h_any, gpu.d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost), "read any_left");
                if (h_any == 0u) {
                    gpuCompleted[gpuIdx] = true;
                } else {
                    gpuSlice[gpuIdx]++;
                    if (gpuSlice[gpuIdx] >= effective_slices_per_launch) gpuSlice[gpuIdx] = 0;
                    std::swap(gpu.d_Px, gpu.d_Rx);
                    std::swap(gpu.d_Py, gpu.d_Ry);
                    gpuNeedsLaunch[gpuIdx] = true;
                }
            } else if (qs == cudaErrorNotReady) {
                any_stream_busy = true;
            } else {
                cudaGetLastError();
                stop_all = true;
                std::cerr << "Warning: GPU " << gpuIdx << " stream error\n";
            }

            unsigned long long h_hashes = 0ull;
            cudaMemcpy(&h_hashes, gpu.d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            gpuHashes[gpuIdx] = h_hashes;
            totalHashes += h_hashes;

            if (!gpuCompleted[gpuIdx]) {
                std::array<uint64_t,4> curKey;
                uint64_t sampleIdx = gpu.threadsTotal > 1 ? (gpu.threadsTotal / 2) : 0;
                cudaMemcpy(curKey.data(), gpu.d_start_scalars + sampleIdx*4, 4*sizeof(uint64_t), cudaMemcpyDeviceToHost);
                gpuCurrentKey[gpuIdx] = curKey;
            }

            int host_found = 0;
            cudaMemcpy(&host_found, gpu.d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
            if (host_found == FOUND_READY) {
                stop_all = true;
                found_any.store(true);
                int found_gpu = gpuIdx;
                FoundResult host_result{};
                cudaMemcpy(&host_result, gpu.d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                drain_partial_results(gpu, gpuIdx);
                std::cout << "\n======== FOUND MATCH! =================================\n";
                std::cout << "Found on GPU " << found_gpu << "\n";
                std::cout << "Private Key   : " << formatHex256(host_result.scalar) << "\n";
                std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
                
                
                std::ofstream out("found_key.txt");
                out << "Private Key: " << formatHex256(host_result.scalar) << "\n";
                out << "Public Key: " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
                out << "GPU: " << found_gpu << "\n";
                out.close();
                std::cout << "Result saved to found_key.txt\n";
                break;
            }
        }

        if (stop_all) break;

        for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
            if (!gpuNeedsLaunch[gpuIdx] || gpuCompleted[gpuIdx]) continue;
            GPUContext& gpu = gpus[gpuIdx];
            cudaSetDevice(gpu.deviceId);

            unsigned int zeroU = 0u;
            ck(cudaMemcpyAsync(gpu.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, gpu.stream), "zero d_any_left");

            cudaError_t launchErr = launch_point_add_and_check_oneinv(gpu, B, effective_slices_per_launch, partial_digits);
            if (launchErr != cudaSuccess) {
                std::cerr << "\nKernel launch error on GPU " << gpuIdx << ": " << cudaGetErrorString(launchErr) << "\n";
                stop_all = true;
            }
            cudaEventRecord(gpu.kernelDone, gpu.stream);
            gpuNeedsLaunch[gpuIdx] = false;
        }

        if (!any_stream_busy && !stop_all && !random_mode) {
            bool all_completed = true;
            for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
                if (!gpuCompleted[gpuIdx]) {
                    all_completed = false;
                    break;
                }
            }
            if (all_completed && cpu_sidecar.active && !cpu_work_taken_over) {
                if (!take_over_cpu_remainder()) stop_all = true;
            } else if (all_completed && !cpu_sidecar.active) {
                stop_all = true;
            }
        }

        if ((any_stream_busy || cpu_sidecar.active) && !stop_all) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_for_limit = std::chrono::duration<double>(now - t0).count();
        if (max_seconds != 0 && elapsed_for_limit >= (double)max_seconds) {
            stop_all = true;
            time_limit_hit = true;
        }
        // --autosavetimer: periodically write the same checkpoint the exit path
        // would, so a crash or a power cut costs at most one interval of work
        // rather than the whole run. Random mode has no linear progress to
        // record, and a run that already found the key is about to delete the
        // checkpoint anyway, so neither autosaves.
        if (autosave_seconds != 0 && !stop_all && !random_mode && !found_any.load() &&
            elapsed_for_limit - last_autosave_time >= (double)autosave_seconds) {
            last_autosave_time = elapsed_for_limit;
            Checkpoint cp;
            if (capture_checkpoint(cp) &&
                write_checkpoint(checkpoint_path, cp, checkpoint_key)) {
                long double cp_done  = ld_from_u256(cp.gpus[0].consumed);
                long double cp_total = ld_from_u256(cp.gpus[0].per_thread);
                std::cout << "\r\n[autosave] checkpoint written to " << checkpoint_path;
                if (cp_total > 0.0L)
                    std::cout << " at " << std::fixed << std::setprecision(2)
                              << (double)(cp_done / cp_total * 100.0L) << "%";
                std::cout << " (" << cp.hashes << " keys checked)\n";
                std::cout.flush();
                autosave_failed_warned = false;
            } else if (!autosave_failed_warned) {
                // Warn once per failure streak: a broken path would otherwise
                // print every interval and bury the status line.
                std::cerr << "\r\nWarning: autosave could not write checkpoint to "
                          << checkpoint_path << "\n";
                autosave_failed_warned = true;
            }
        }
        if (random_mode && !stop_all &&
            elapsed_for_limit - last_random_time >= (double)random_interval_seconds) {
            if (!rerandomize_all_gpus()) {
                stop_all = true;
            }
        }
        double dt = std::chrono::duration<double>(now - tLast).count();
        if (dt >= 1.0) {
            double delta = (double)(totalHashes - lastHashes);
            double mkeys = delta / (dt * 1e6);
            double elapsed = std::chrono::duration<double>(now - t0).count();
            bool have_cpu_stats = cpu_sidecar_enabled && read_cpu_live_stats(cpu_sidecar.stats_path, cpuLiveStats);
            bool cpu_speed_active = have_cpu_stats && cpu_sidecar.active && !cpuLiveStats.done;
            double cpu_live_mkeys = cpu_speed_active ? cpuLiveStats.mkeys : 0.0;
            unsigned long long cpu_live_checked = have_cpu_stats ? cpuLiveStats.checked : 0ull;
            if (cpu_random_mode && !cpu_sidecar.active) cpu_live_checked = 0ull;
            unsigned long long displayCount = totalHashes + resume_hashes + cpu_random_checked_base + cpu_live_checked;
            double displayMkeys = mkeys + cpu_live_mkeys;
            long double total_keys_ld = ld_from_u256(full_range_len);
            long double prog = total_keys_ld > 0.0L ? ((long double)displayCount / total_keys_ld) * 100.0L : 0.0L;
            if (prog > 100.0L) prog = 100.0L;

            std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                      << " s | Speed: " << std::fixed << std::setprecision(1) << displayMkeys
                      << " Mkeys/s";
            if (have_cpu_stats) {
                std::cout << " | GPU: " << std::fixed << std::setprecision(1) << mkeys
                          << " | CPU: " << std::fixed << std::setprecision(1) << cpu_live_mkeys;
            }
            std::cout << " | Count: " << displayCount
                      << " | Progress: " << std::fixed << std::setprecision(2) << (double)prog << " %";
            
            for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
                if (gpuCompleted[gpuIdx]) std::cout << " [GPU" << gpuIdx << ":done]";
                else std::cout << " [GPU" << gpuIdx << ":S" << (gpuSlice[gpuIdx]+1) << "/" << effective_slices_per_launch << "|" << formatHex256Trimmed(gpuCurrentKey[gpuIdx].data()) << "]";
            }
            if (have_cpu_stats) {
                std::cout << " [CPU:" << cpuLiveStats.threads << "T"
                          << "|" << std::fixed << std::setprecision(2) << cpuLiveStats.progress << "%";
                if (cpuLiveStats.done) std::cout << "|done";
                std::cout << "]";
            }
            print_partial_live_counts();
            std::cout.flush();
            lastHashes = totalHashes; tLast = now;

        }

        if (!any_stream_busy && !stop_all && !random_mode) {
            bool all_completed = true;
            for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
                if (!gpuCompleted[gpuIdx]) {
                    all_completed = false;
                    break;
                }
            }
            if (all_completed && cpu_sidecar.active && !cpu_work_taken_over) {
                if (!take_over_cpu_remainder()) stop_all = true;
            } else if (all_completed && !cpu_sidecar.active) {
                stop_all = true;
            }
        }
    }

    if (cpu_sidecar.active) {
        terminate_cpu_sidecar(cpu_sidecar);
    }

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        GPUContext& gpu = gpus[gpuIdx];
        cudaSetDevice(gpu.deviceId);
        cudaStreamSynchronize(gpu.stream);
        drain_partial_results(gpu, gpuIdx);
    }
    std::cout << "\n";

    if (found_any.load() && partial_digits != 0) {
        std::cout << "======== PARTIALS SAVED ===============================\n";
        bool any_partial_saved = false;
        for (size_t extra = 0; extra < partialSavedByExtra.size(); ++extra) {
            unsigned long long saved = partialSavedByExtra[extra];
            if (saved == 0ull) continue;
            any_partial_saved = true;
            if (extra == 0) {
                std::cout << "partial.txt: " << saved << "\n";
            } else {
                std::cout << "partialp" << extra << ".txt: " << saved << "\n";
            }
        }
        if (!any_partial_saved) {
            std::cout << "No partial candidates were saved before the match.\n";
        }
    }

    int exit_code = EXIT_SUCCESS;

    if (!found_any.load()) {
        if (g_sigint) {
            std::cout << "======== INTERRUPTED (Ctrl+C) ==========================\n";
            std::cout << "Search was interrupted by user. Partial progress above.\n";
            exit_code = 130;
        } else if (time_limit_hit) {
            std::cout << "======== STOPPED (time limit) =========================\n";
            std::cout << "Reached --seconds " << max_seconds
                      << " before the range was exhausted; it was NOT fully searched.\n";
            exit_code = 2;
        } else if (random_mode) {
            std::cout << "======== KEY NOT FOUND (random mode) ==================\n";
            std::cout << "Target hash160 was not found in random sweeps.\n";
        } else {
            std::cout << "======== KEY NOT FOUND (exhaustive) ===================\n";
            std::cout << "Target hash160 was not found within the specified range.\n";
        }
    }


    // A run cut short by Ctrl+C or --seconds writes a checkpoint so the same
    // command plus --resume continues from here. A finished or successful run
    // has nothing to resume, and random mode has no linear progress to record --
    // in those cases clear any stale file instead, so a later --resume cannot
    // silently restart a range that is already done.
    const bool stopped_early = !found_any.load() && !random_mode && (g_sigint || time_limit_hit);
    if (stopped_early) {
        Checkpoint cp;
        const bool ok = capture_checkpoint(cp);

        if (ok && write_checkpoint(checkpoint_path, cp, checkpoint_key)) {
            long double cp_done  = ld_from_u256(cp.gpus[0].consumed);
            long double cp_total = ld_from_u256(cp.gpus[0].per_thread);
            std::cout << "Checkpoint saved to " << checkpoint_path;
            if (cp_total > 0.0L)
                std::cout << " at " << std::fixed << std::setprecision(2)
                          << (double)(cp_done / cp_total * 100.0L) << "%";
            std::cout << " (" << cp.hashes << " keys checked, encrypted)\n";
            std::cout << "Resume with the same command plus --resume";
            if (checkpoint_path != "cyclone_checkpoint.txt")
                std::cout << " --checkpoint " << checkpoint_path;
            std::cout << "\n";
            if (!cp.cpu_tail.empty())
                std::cout << "Note: only GPU progress is saved; the CPU sidecar tail "
                          << cp.cpu_tail << " will restart from its beginning.\n";
        } else {
            std::cerr << "Warning: could not write checkpoint to " << checkpoint_path << "\n";
        }
    } else if (!g_sigint) {
        std::remove(checkpoint_path.c_str());
    }

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        GPUContext& gpu = gpus[gpuIdx];
        cudaSetDevice(gpu.deviceId);
        cudaFree(gpu.d_start_scalars); cudaFree(gpu.d_Px); cudaFree(gpu.d_Py);
        cudaFree(gpu.d_Rx); cudaFree(gpu.d_Ry); cudaFree(gpu.d_counts256);
        cudaFree(gpu.d_found_flag); cudaFree(gpu.d_found_result);
        cudaFree(gpu.d_partial_results); cudaFree(gpu.d_partial_count); cudaFree(gpu.d_partial_overflow);
        cudaFree(gpu.d_hashes_accum); cudaFree(gpu.d_any_left);
        if (gpu.h_start_scalars) cudaFreeHost(gpu.h_start_scalars);
        if (gpu.h_counts256) cudaFreeHost(gpu.h_counts256);
        cudaEventDestroy(gpu.kernelDone);
        cudaStreamDestroy(gpu.stream);
    }

    return exit_code;
}
