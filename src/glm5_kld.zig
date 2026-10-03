//! Gated native GLM teacher-forced KLD; no public serving registration.
const std = @import("std");
const mlx = @import("mlx.zig");
const kld = @import("kld.zig");
const model = @import("model.zig");
const native = @import("glm5_diagnostic.zig");
const forward = @import("glm5_forward.zig");
const readExact = @import("expert_stream.zig").readExact;
const Arr = mlx.mlx_array;
const Hash = std.crypto.hash.sha2.Sha256;
pub const schema = "sushi-glm-native-kld-v1";

pub const Metrics = struct {
    positions: usize,
    kld: f64,
    top1_matches: usize,
    top1_rate: f64,
    nll: f64,
    cosine_similarity: f64,
    cosine_loss: f64,
    worst_cosine_loss: f64,
};
pub const Totals = struct {
    positions: usize = 0,
    kld: f64 = 0,
    top1: usize = 0,
    nll: f64 = 0,
    cosine: f64 = 0,
    cosine_loss: f64 = 0,
    worst: f64 = 0,
    pub fn add(self: *Totals, score: kld.RowScore) void {
        self.positions += 1;
        self.kld += score.kld;
        self.nll += score.nll;
        self.top1 += @intFromBool(score.top1);
        self.cosine += score.cosine_similarity;
        self.cosine_loss += score.cosine_loss;
        self.worst = @max(self.worst, score.cosine_loss);
    }
    pub fn merge(self: *Totals, other: Totals) void {
        self.positions += other.positions;
        self.kld += other.kld;
        self.nll += other.nll;
        self.top1 += other.top1;
        self.cosine += other.cosine;
        self.cosine_loss += other.cosine_loss;
        self.worst = @max(self.worst, other.worst);
    }
    pub fn metrics(self: Totals) Metrics {
        const n: f64 = @floatFromInt(@max(self.positions, 1));
        return .{ .positions = self.positions, .kld = self.kld / n, .top1_matches = self.top1, .top1_rate = @as(f64, @floatFromInt(self.top1)) / n, .nll = self.nll / n, .cosine_similarity = self.cosine / n, .cosine_loss = self.cosine_loss / n, .worst_cosine_loss = self.worst };
    }
};
pub const Scored = struct {
    all: Totals = .{},
    through_eos: Totals = .{},
    first_eos: ?usize,
    per_position_kld: []f64,
    teacher_sha256: [64]u8 = undefined,
    pub fn deinit(self: Scored, a: std.mem.Allocator) void {
        a.free(self.per_position_kld);
    }
};
fn openRead(a: std.mem.Allocator, path: []const u8) !std.c.fd_t {
    const z = try a.dupeSentinel(u8, path, 0);
    defer a.free(z);
    const fd = std.c.open(z.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.GlmKldFileMissing;
    return fd;
}
fn fileSize(fd: std.c.fd_t) !u64 {
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0 or st.size < 0) return error.GlmKldFileStat;
    return @intCast(st.size);
}
fn digest(raw: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    Hash.hash(raw, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}
fn hashFile(a: std.mem.Allocator, path: []const u8) ![64]u8 {
    const fd = try openRead(a, path);
    defer _ = std.c.close(fd);
    const size = try fileSize(fd);
    var h = Hash.init(.{});
    var buffer: [64 * 1024]u8 = undefined;
    var at: u64 = 0;
    while (at < size) {
        const n: usize = @intCast(@min(buffer.len, size - at));
        try readExact(fd, buffer[0..n], at);
        h.update(buffer[0..n]);
        at += n;
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return std.fmt.bytesToHex(out, .lower);
}
fn atomicJson(a: std.mem.Allocator, io: std.Io, path: []const u8, value: anytype) !void {
    const raw = try std.json.Stringify.valueAlloc(a, value, .{ .whitespace = .indent_2 });
    defer a.free(raw);
    const temp = try std.fmt.allocPrintSentinel(a, "{s}.tmp", .{path}, 0);
    defer a.free(temp);
    const final = try a.dupeSentinel(u8, path, 0);
    defer a.free(final);
    errdefer _ = std.c.unlink(temp.ptr);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = temp, .data = raw });
    if (std.c.rename(temp.ptr, final.ptr) != 0) return error.GlmKldRename;
}
pub fn copyLogits(s: mlx.mlx_stream, logits: Arr, destination: []f32) !void {
    if (mlx.mlx_array_size(logits) != destination.len) return error.KldShapeMismatch;
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_astype(&wide, logits, .float32, s));
    try mlx.check(mlx.mlx_array_eval(wide));
    @memcpy(destination, (mlx.mlx_array_data_float32(wide) orelse return error.MlxArrayDataNull)[0..destination.len]);
}
/// Row zero predicts generated[0] after the prompt. Row p>0 consumes generated[p-1].
/// Last teacher token is scored, not forwarded; request offset ends at P+G-1.
pub fn scorePrompt(a: std.mem.Allocator, net: *const forward.Model, request: *forward.Request, prompt: []const u32, generated: []const u32, fd: std.c.fd_t, eos: []const u32, chunk: usize) !Scored {
    if (prompt.len == 0 or generated.len == 0 or chunk == 0 or chunk > 65536) return error.EmptyFixturePrompt;
    if (request.offset != 0 or request.failed or request.capture != null) return error.GlmKldRequiresEmptyRequest;
    const vocab: usize = net.cfg.vocab_size;
    const teacher = try a.alloc(f32, vocab);
    defer a.free(teacher);
    const student = try a.alloc(f32, vocab);
    defer a.free(student);
    var result = Scored{ .first_eos = kld.firstEosPosition(generated, eos), .per_position_kld = try a.alloc(f64, generated.len) };
    errdefer result.deinit(a);
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    var at: usize = 0;
    while (at < prompt.len) {
        const end = @min(prompt.len, at + chunk);
        const input = mlx.mlx_array_new_data(prompt[at..end].ptr, &[_]c_int{ 1, @intCast(end - at) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(input);
        const next = try net.forwardLast(request, input, true);
        _ = mlx.mlx_array_free(logits);
        logits = next;
        at = end;
    }
    var teacher_hash = Hash.init(.{});
    const through = if (result.first_eos) |position| position + 1 else generated.len;
    for (generated, 0..) |token, position| {
        if (position > 0) {
            const input = mlx.mlx_array_new_data(&generated[position - 1], &[_]c_int{ 1, 1 }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const next = try net.forwardLast(request, input, true);
            _ = mlx.mlx_array_free(logits);
            logits = next;
        }
        try readExact(fd, std.mem.sliceAsBytes(teacher), position * vocab * @sizeOf(f32));
        teacher_hash.update(std.mem.sliceAsBytes(teacher));
        try copyLogits(net.s, logits, student);
        const row = try kld.scoreRow(teacher, student, token);
        result.all.add(row);
        if (position < through) result.through_eos.add(row);
        result.per_position_kld[position] = row.kld;
    }
    var teacher_digest: [32]u8 = undefined;
    teacher_hash.final(&teacher_digest);
    result.teacher_sha256 = std.fmt.bytesToHex(teacher_digest, .lower);
    return result;
}
const Prompt = struct {
    ids: []u32,
    generated: []u32,
    fd: std.c.fd_t,
    prompt_sha256: [64]u8,
    generated_sha256: [64]u8,
    logits_sha256: [64]u8,
    fn deinit(self: Prompt, a: std.mem.Allocator) void {
        a.free(self.ids);
        a.free(self.generated);
        _ = std.c.close(self.fd);
    }
};
fn safeRelativePromptDir(path: []const u8) bool {
    if (path.len == 0 or path[0] == '/' or std.mem.indexOfAny(u8, path, "\\\x00") != null) return false;
    var components = std.mem.splitScalar(u8, path, '/');
    while (components.next()) |part| if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    return true;
}
fn validateManifestComplete(a: std.mem.Allocator, io: std.Io, fixture: []const u8) !void {
    const path = try std.fmt.allocPrint(a, "{s}/baseline.json", .{fixture});
    defer a.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 * 1024 * 1024));
    defer a.free(raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.GlmTeacherIncomplete;
    const complete = parsed.value.object.get("complete") orelse return error.GlmTeacherIncomplete;
    if (complete != .bool or !complete.bool) return error.GlmTeacherIncomplete;
}
const Study = struct {
    requested_prompt_count: usize = 4,
    completed_prompt_count: usize,
    actual_positions: usize,
    truncated: bool = false,
    stopped_by_user: bool = false,
    stop_reason: ?[]const u8 = null,
    fn deinit(self: Study, a: std.mem.Allocator) void {
        if (self.stop_reason) |reason| a.free(reason);
    }
};
fn studyContract(a: std.mem.Allocator, value: std.json.Value, count: usize) !Study {
    if (value != .object) return error.BadBaselineJson;
    if (count == 4) {
        if (value.object.get("truncated")) |flag| if (flag != .bool or flag.bool) return error.BadBaselineJson;
        return .{ .completed_prompt_count = 4, .actual_positions = 2048 };
    }
    if ((count != 1 and count != 2) or try jsonInt(value, "requested_prompt_count") != 4 or try jsonInt(value, "completed_prompt_count") != count or try jsonInt(value, "actual_positions") != count * 512) return error.BadBaselineJson;
    for ([_][]const u8{ "truncated", "stopped_by_user" }) |key| {
        const flag = value.object.get(key) orelse return error.BadBaselineJson;
        if (flag != .bool or !flag.bool) return error.BadBaselineJson;
    }
    const reason = value.object.get("stop_reason") orelse return error.BadBaselineJson;
    if (reason != .string or reason.string.len == 0 or reason.string.len > 4096) return error.BadBaselineJson;
    if (count == 1 and !std.mem.eql(u8, reason.string, "user_requested_one_completed_prompt")) return error.BadBaselineJson;
    return .{ .completed_prompt_count = count, .actual_positions = count * 512, .truncated = true, .stopped_by_user = true, .stop_reason = try a.dupe(u8, reason.string) };
}
fn readStudy(a: std.mem.Allocator, io: std.Io, fixture: []const u8, count: usize) !Study {
    const path = try std.fmt.allocPrint(a, "{s}/baseline.json", .{fixture});
    defer a.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 * 1024 * 1024));
    defer a.free(raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer parsed.deinit();
    return studyContract(a, parsed.value, count);
}
fn validatePrompt(a: std.mem.Allocator, io: std.Io, root: []const u8, record: kld.FixturePrompt, vocab: usize, max_context: usize) !Prompt {
    if (record.id.len == 0 or !safeRelativePromptDir(record.dir)) return error.BadBaselineJson;
    const prefix = try std.fmt.allocPrint(a, "{s}/{s}", .{ root, record.dir });
    defer a.free(prefix);
    const pp = try std.fmt.allocPrint(a, "{s}/prompt_tokens.txt", .{prefix});
    defer a.free(pp);
    const gp = try std.fmt.allocPrint(a, "{s}/generated_tokens.txt", .{prefix});
    defer a.free(gp);
    const lp = try std.fmt.allocPrint(a, "{s}/logits.f32", .{prefix});
    defer a.free(lp);
    const ids = try kld.readIdList(a, io, pp);
    errdefer a.free(ids);
    const generated = try kld.readIdList(a, io, gp);
    errdefer a.free(generated);
    if (ids.len == 0 or generated.len == 0 or ids.len != record.prompt_tokens or generated.len != record.generated_tokens) return error.GlmKldTokenCountMismatch;
    if (ids.len > max_context or generated.len > max_context - ids.len) return error.GlmContextExceeded;
    for (ids) |id| if (id >= vocab) return error.InvalidGlmDraftToken;
    for (generated) |id| if (id >= vocab) return error.InvalidGlmDraftToken;
    const expected = try std.math.mul(u64, try std.math.mul(u64, generated.len, vocab), 4);
    const fd = try openRead(a, lp);
    errdefer _ = std.c.close(fd);
    if (try fileSize(fd) != expected) return error.TeacherLogitsSizeMismatch;
    const row = try a.alloc(f32, vocab);
    defer a.free(row);
    var hash = Hash.init(.{});
    for (generated, 0..) |token, position| {
        try readExact(fd, std.mem.sliceAsBytes(row), position * vocab * 4);
        hash.update(std.mem.sliceAsBytes(row));
        var best: usize = 0;
        var norm: f64 = 0;
        for (row, 0..) |v, i| {
            if (!std.math.isFinite(v)) return error.NonFinite;
            if (v > row[best]) best = i;
            norm += @as(f64, v) * v;
        }
        if (norm == 0) return error.ZeroNorm;
        if (best != token) return error.GlmTeacherRowTokenMismatch;
    }
    var h: [32]u8 = undefined;
    hash.final(&h);
    return .{ .ids = ids, .generated = generated, .fd = fd, .prompt_sha256 = try hashFile(a, pp), .generated_sha256 = try hashFile(a, gp), .logits_sha256 = std.fmt.bytesToHex(h, .lower) };
}
fn shardIdentity(a: std.mem.Allocator, io: std.Io, path: []const u8, index_path: []const u8) ![64]u8 {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, index_path, a, .limited(16 * 1024 * 1024));
    defer a.free(raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidGlmWeightIndex;
    const wm = parsed.value.object.get("weight_map") orelse return error.InvalidGlmWeightIndex;
    if (wm != .object) return error.InvalidGlmWeightIndex;
    var seen = std.StringHashMap(void).init(a);
    defer seen.deinit();
    var hash = Hash.init(.{});
    for (wm.object.values()) |value| {
        if (value != .string or value.string.len == 0 or std.mem.indexOfAny(u8, value.string, "/\\") != null or std.mem.eql(u8, value.string, "..")) return error.InvalidGlmShardName;
        if (seen.contains(value.string)) continue;
        try seen.put(value.string, {});
        const file = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ path, value.string }, 0);
        defer a.free(file);
        var st: std.c.Stat = undefined;
        if (std.c.stat(file.ptr, &st) != 0 or st.size < 0) return error.GlmKldFileStat;
        const mt = st.mtime();
        const entry = try std.fmt.allocPrint(a, "{s}:{d}:{d}:{d}:{d}\n", .{ value.string, st.ino, st.size, mt.sec, mt.nsec });
        defer a.free(entry);
        hash.update(entry);
    }
    var out: [32]u8 = undefined;
    hash.final(&out);
    return std.fmt.bytesToHex(out, .lower);
}
fn jsonInt(value: std.json.Value, key: []const u8) !usize {
    if (value != .object) return error.BadBaselineJson;
    const v = value.object.get(key) orelse return error.BadBaselineJson;
    if (v != .integer or v.integer < 0) return error.BadBaselineJson;
    return @intCast(v.integer);
}
fn env(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}
fn completedMatches(a: std.mem.Allocator, raw: []const u8, fingerprint: []const u8, prompt_count: usize, positions: usize) !bool {
    const parsed = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const obj = parsed.value.object;
    const complete = obj.get("complete") orelse return false;
    const saved = obj.get("fingerprint") orelse return false;
    const version = obj.get("schema") orelse return false;
    const prompts = obj.get("prompts") orelse return false;
    const all = obj.get("all_positions") orelse return false;
    if (complete != .bool or !complete.bool or saved != .string or !std.mem.eql(u8, saved.string, fingerprint) or version != .string or !std.mem.eql(u8, version.string, schema) or prompts != .array or prompts.array.items.len != prompt_count) return false;
    if ((jsonInt(all, "positions") catch return false) != positions) return false;
    var sum: usize = 0;
    for (prompts.array.items) |entry| {
        if (entry != .object) return false;
        const scores = entry.object.get("per_position_kld") orelse return false;
        const metrics = entry.object.get("all_positions") orelse return false;
        const through = entry.object.get("through_first_eos") orelse return false;
        const count = jsonInt(metrics, "positions") catch return false;
        const eos_count = jsonInt(through, "positions") catch return false;
        if (scores != .array or scores.array.items.len != count or count == 0 or eos_count == 0 or eos_count > count) return false;
        sum = std.math.add(usize, sum, count) catch return false;
    }
    return sum == positions;
}
const CategoryReport = struct { all_positions: Metrics, through_first_eos: Metrics, prompts: usize };
const PromptReport = struct { id: []const u8, category: []const u8, prompt_tokens: usize, first_eos_position: ?usize, all_positions: Metrics, through_first_eos: Metrics, per_position_kld: []const f64, prompt_sha256: []const u8, generated_sha256: []const u8, logits_sha256: []const u8 };

test "GLM KLD real teacher comparison" {
    const path = env("SUSHI_GLM_KLD_MODEL") orelse return error.SkipZigTest;
    const fixture = env("SUSHI_GLM_KLD_FIXTURE") orelse return error.MissingGlmKldFixture;
    const output = env("SUSHI_GLM_KLD_OUT") orelse return error.MissingGlmKldOutput;
    const revision = env("SUSHI_GLM_KLD_SOURCE_REV") orelse return error.MissingGlmKldIdentity;
    const binary_sha = env("SUSHI_GLM_KLD_BINARY_SHA256") orelse return error.MissingGlmKldIdentity;
    if (binary_sha.len != 64) return error.MissingGlmKldIdentity;
    for (binary_sha) |byte| if (!std.ascii.isHex(byte)) return error.MissingGlmKldIdentity;
    if (!std.mem.eql(u8, env("MLX_ENABLE_TF32") orelse "", "0")) return error.GlmKldRequiresTf32Disabled;
    const a = std.testing.allocator;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var cfg = try model.parseConfig(io, a, path);
    defer cfg.deinit(a);
    if (!cfg.isGlm5()) return error.InvalidGlmConfig;
    try validateManifestComplete(a, io, fixture);
    var baseline = try kld.readBaseline(a, io, fixture);
    defer baseline.deinit();
    if (!std.mem.eql(u8, baseline.schema, kld.SCHEMA) or baseline.tokens_per_prompt != 512) return error.BadBaselineJson;
    const study = try readStudy(a, io, fixture, baseline.prompts.len);
    defer study.deinit(a);
    const identity_path = try std.fmt.allocPrint(a, "{s}/identity.json", .{fixture});
    defer a.free(identity_path);
    const identity_raw = try std.Io.Dir.cwd().readFileAlloc(io, identity_path, a, .limited(4 * 1024 * 1024));
    defer a.free(identity_raw);
    const identity = try std.json.parseFromSlice(std.json.Value, a, identity_raw, .{});
    defer identity.deinit();
    const chunk = try jsonInt(identity.value, "prefix_chunk");
    if (chunk != 512 or try jsonInt(identity.value, "tokens_per_prompt") != 512) return error.BadBaselineJson;
    for ([_]struct { key: []const u8, value: []const u8 }{ .{ .key = "kv_cache_format", .value = "bf16" }, .{ .key = "kda_state_format", .value = "float32" } }) |field| {
        const v = identity.value.object.get(field.key) orelse return error.BadBaselineJson;
        if (v != .string or !std.mem.eql(u8, v.string, field.value)) return error.BadBaselineJson;
    }
    const prompts = try a.alloc(Prompt, baseline.prompts.len);
    defer a.free(prompts);
    var made: usize = 0;
    defer for (prompts[0..made]) |p| p.deinit(a);
    var code_prompts: usize = 0;
    var prose_prompts: usize = 0;
    for (baseline.prompts, 0..) |record, i| {
        if (std.mem.startsWith(u8, record.id, "code-")) code_prompts += 1 else if (std.mem.startsWith(u8, record.id, "prose-")) prose_prompts += 1 else return error.BadBaselineJson;
        if (record.generated_tokens != 512) return error.GlmKldTokenCountMismatch;
        for (baseline.prompts[0..i]) |prior| if (std.mem.eql(u8, record.id, prior.id) or std.mem.eql(u8, record.dir, prior.dir)) return error.BadBaselineJson;
        prompts[i] = try validatePrompt(a, io, fixture, record, cfg.vocab_size, cfg.max_position_embeddings);
        made += 1;
    }
    if (code_prompts != (if (study.truncated) study.completed_prompt_count else 2) or prose_prompts != (if (study.truncated) @as(usize, 0) else 2)) return error.BadBaselineJson;
    if (study.completed_prompt_count == 1 and !std.mem.eql(u8, baseline.prompts[0].id, "code-python-topological-sort")) return error.BadBaselineJson;
    const config_path = try std.fmt.allocPrint(a, "{s}/config.json", .{path});
    defer a.free(config_path);
    const index_path = try std.fmt.allocPrint(a, "{s}/model.safetensors.index.json", .{path});
    defer a.free(index_path);
    const baseline_path = try std.fmt.allocPrint(a, "{s}/baseline.json", .{fixture});
    defer a.free(baseline_path);
    const config_sha = try hashFile(a, config_path);
    const index_sha = try hashFile(a, index_path);
    const baseline_sha = try hashFile(a, baseline_path);
    const identity_sha = digest(identity_raw);
    const tokenizer_path = try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{path});
    defer a.free(tokenizer_path);
    const tokenizer_sha = try hashFile(a, tokenizer_path);
    const teacher_tokenizer = identity.value.object.get("tokenizer_sha256") orelse return error.BadBaselineJson;
    if (teacher_tokenizer != .string or !std.mem.eql(u8, &tokenizer_sha, teacher_tokenizer.string)) return error.GlmKldTokenizerMismatch;
    const shards_sha = try shardIdentity(a, io, path, index_path);
    var fingerprint_hash = Hash.init(.{});
    for ([_][]const u8{ path, fixture, revision, binary_sha, &config_sha, &index_sha, &tokenizer_sha, &baseline_sha, &identity_sha, &shards_sha }) |part| {
        fingerprint_hash.update(part);
        fingerprint_hash.update(&.{0});
    }
    for (prompts) |p| {
        fingerprint_hash.update(&p.prompt_sha256);
        fingerprint_hash.update(&p.generated_sha256);
        fingerprint_hash.update(&p.logits_sha256);
    }
    const variant = .{ .lane_pair = env("SUSHI_GLM_LANE_PAIR") orelse "default", .down_lane = env("SUSHI_GLM_DOWN_LANE") orelse "default", .hc_prefill = env("SUSHI_GLM_HC_PREFILL") orelse "default", .hc_serial = env("SUSHI_GLM_HC_FUSED") orelse "default", .middle = env("SUSHI_EXL3_CLAMPED_MIDDLE") orelse "default" };
    const variant_raw = try std.json.Stringify.valueAlloc(a, variant, .{});
    defer a.free(variant_raw);
    fingerprint_hash.update(variant_raw);
    var fingerprint_bytes: [32]u8 = undefined;
    fingerprint_hash.final(&fingerprint_bytes);
    const fingerprint = std.fmt.bytesToHex(fingerprint_bytes, .lower);
    // Only a complete, identical result can be reused; partial model state is never resumed.

    const existing: ?[]u8 = std.Io.Dir.cwd().readFileAlloc(io, output, a, .limited(32 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    if (existing) |raw| {
        defer a.free(raw);
        var positions: usize = 0;
        for (prompts) |p| positions += p.generated.len;
        if (!try completedMatches(a, raw, &fingerprint, prompts.len, positions)) return error.GlmKldOutputExists;
        return;
    }
    const progress = try std.fmt.allocPrint(a, "{s}.progress.json", .{output});
    defer a.free(progress);
    try atomicJson(a, io, progress, .{ .complete = false, .phase = "validated", .prompts = prompts.len });
    try @import("mimo_source.zig").validateExl3Pack(io, a, path, &cfg);
    var tok = try @import("tokenizer.zig").loadTokenizer(io, a, path);
    defer tok.deinit();
    var eos: std.ArrayList(u32) = .empty;
    defer eos.deinit(a);
    try eos.appendSlice(a, cfg.eosTokenSlice());
    if (tok.encode(a, "<|im_end|>")) |ids| {
        defer a.free(ids);
        if (ids.len == 1 and std.mem.indexOfScalar(u32, eos.items, ids[0]) == null) try eos.append(a, ids[0]);
    } else |_| {}
    const limit: usize = 110 * 1024 * 1024 * 1024;
    var old_memory: usize = 0;
    var old_cache: usize = 0;
    var old_wired: usize = 0;
    var ignored: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_memory, limit));
    defer _ = mlx.mlx_set_memory_limit(&ignored, old_memory);
    try mlx.check(mlx.mlx_set_cache_limit(&old_cache, 2 * 1024 * 1024 * 1024));
    defer _ = mlx.mlx_set_cache_limit(&ignored, old_cache);
    const wired = @min(limit, mlx.maxRecommendedWorkingSet());
    try mlx.check(mlx.mlx_set_wired_limit(&old_wired, wired));
    defer _ = mlx.mlx_set_wired_limit(&ignored, old_wired);
    try mlx.check(mlx.mlx_reset_peak_memory());
    var weights = try native.loadWeights(io, a, path, s);
    defer weights.deinit();
    var net = try forward.Model.load(a, cfg, &weights, s);
    defer net.deinit();
    var request = try forward.Request.init(a, cfg.num_hidden_layers);
    defer request.deinit();
    request.dense_prefill = true;
    request.prefill_async = true;
    request.prefill_sync_layers = 2;
    request.decode_async = true;
    const scores = try a.alloc(Scored, prompts.len);
    defer a.free(scores);
    var scored: usize = 0;
    defer for (scores[0..scored]) |value| value.deinit(a);
    const reports = try a.alloc(PromptReport, prompts.len);
    defer a.free(reports);
    var all: Totals = .{};
    var through: Totals = .{};
    var category_all: [2]Totals = @splat(.{});
    var category_eos: [2]Totals = @splat(.{});
    const timer = @import("io_util.zig").Stopwatch.init(io);
    for (prompts, baseline.prompts, 0..) |p, record, i| {
        request.reset();
        try atomicJson(a, io, progress, .{ .complete = false, .phase = "scoring", .prompt = record.id, .completed_prompts = i });
        scores[i] = try scorePrompt(a, &net, &request, p.ids, p.generated, p.fd, eos.items, chunk);
        scored += 1;
        const value = scores[i];
        if (!std.mem.eql(u8, &value.teacher_sha256, &p.logits_sha256)) return error.GlmKldSourceChanged;
        all.merge(value.all);
        through.merge(value.through_eos);
        const category: usize = if (std.mem.startsWith(u8, record.id, "code-")) 0 else 1;
        category_all[category].merge(value.all);
        category_eos[category].merge(value.through_eos);
        reports[i] = .{ .id = record.id, .category = if (category == 0) "code" else "prose", .prompt_tokens = p.ids.len, .first_eos_position = value.first_eos, .all_positions = value.all.metrics(), .through_first_eos = value.through_eos.metrics(), .per_position_kld = value.per_position_kld, .prompt_sha256 = &prompts[i].prompt_sha256, .generated_sha256 = &prompts[i].generated_sha256, .logits_sha256 = &prompts[i].logits_sha256 };
    }
    if (all.positions != study.actual_positions) return error.GlmKldTokenCountMismatch;
    var peak: usize = 0;
    var active: usize = 0;
    var cached: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    try mlx.check(mlx.mlx_get_active_memory(&active));
    try mlx.check(mlx.mlx_get_cache_memory(&cached));
    const final_shards = try shardIdentity(a, io, path, index_path);
    if (!std.mem.eql(u8, &shards_sha, &final_shards)) return error.GlmKldSourceChanged;
    try atomicJson(a, io, output, .{ .schema = schema, .complete = true, .student = path, .teacher = baseline.model, .fixture = fixture, .fingerprint = &fingerprint, .source_revision = revision, .binary_sha256 = binary_sha, .student_config_sha256 = &config_sha, .student_index_sha256 = &index_sha, .student_tokenizer_sha256 = &tokenizer_sha, .student_shard_stat_sha256 = &shards_sha, .variant = variant, .baseline_sha256 = &baseline_sha, .teacher_identity_sha256 = &identity_sha, .teacher_identity = identity.value, .teacher_study = study, .quality_scope = if (study.completed_prompt_count == 1) "one completed code prompt; no prose; explicitly user-truncated from four requested prompts; not a release verdict; teacher/student engine floor unmeasured" else if (study.truncated) "two completed code prompts; no prose; explicitly user-truncated from four requested prompts; not a release verdict; teacher/student engine floor unmeasured" else "four mixed prompts (two code/two prose); not the standard sixteen-prompt release verdict; teacher/student engine floor unmeasured", .kv = "BF16 compressed MLA cache; FP32 KDA state", .vocab_size = cfg.vocab_size, .teacher_logits = "full-vocabulary little-endian float32", .student_logits = "native logits widened to float32 without requantization", .teacher_forcing = true, .mtp = false, .dflash = false, .dense_prefill = true, .prefill_chunk = chunk, .prefill_async_layers = 2, .decode_async_layers = 4, .tf32 = false, .eos_ids = eos.items, .eos_inclusive = true, .all_positions = all.metrics(), .through_first_eos = through.metrics(), .categories = .{ .code = CategoryReport{ .all_positions = category_all[0].metrics(), .through_first_eos = category_eos[0].metrics(), .prompts = code_prompts }, .prose = if (prose_prompts == 0) @as(?CategoryReport, null) else CategoryReport{ .all_positions = category_all[1].metrics(), .through_first_eos = category_eos[1].metrics(), .prompts = prose_prompts } }, .prompts = reports, .score_seconds = @as(f64, @floatFromInt(timer.read())) / 1e9, .memory = .{ .peak_bytes = peak, .active_bytes = active, .cache_bytes = cached, .stored_tensor_bytes = native.storedBytes(&weights), .limit_bytes = limit, .wired_bytes = wired }, .partial_resume = false });
    try atomicJson(a, io, progress, .{ .complete = true, .phase = "complete", .positions = all.positions });
}

test "GLM KLD weighted aggregation and inclusive EOS" {
    const a = try kld.scoreRow(&.{ 1, 3, -1 }, &.{ 1, 3, -1 }, 1);
    const b = try kld.scoreRow(&.{ 2, -1, 0 }, &.{ 1, 2, 0 }, 0);
    var left: Totals = .{};
    left.add(a);
    var right: Totals = .{};
    right.add(b);
    right.add(b);
    left.merge(right);
    try std.testing.expectApproxEqAbs(b.kld * 2 / 3, left.metrics().kld, 1e-12);
    try std.testing.expectEqual(@as(usize, 3), left.positions);
    try std.testing.expectEqual(@as(usize, 1), left.top1);
    try std.testing.expectEqual(@as(?usize, 1), kld.firstEosPosition(&.{ 2, 9, 4, 9 }, &.{9}));
}

test "GLM KLD preflight rejects incomplete or misaligned fixtures before inference" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "prompts", .default_dir);
    try tmp.dir.createDir(io, "prompts/00_code-python-topological-sort", .default_dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "prompts/00_code-python-topological-sort/prompt_tokens.txt", .data = "0,2" });
    try tmp.dir.writeFile(io, .{ .sub_path = "prompts/00_code-python-topological-sort/generated_tokens.txt", .data = "1,0" });
    const rows = [_]f32{ 1, 3, -1, 4, 2, 0 };
    try tmp.dir.writeFile(io, .{ .sub_path = "prompts/00_code-python-topological-sort/logits.f32", .data = std.mem.sliceAsBytes(&rows) });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = buf[0..try tmp.dir.realPath(io, &buf)];
    const record = kld.FixturePrompt{ .id = @constCast("fixture"), .dir = @constCast("prompts/00_code-python-topological-sort"), .prompt_tokens = 2, .generated_tokens = 2 };
    const valid = try validatePrompt(a, io, path, record, 3, 8);
    valid.deinit(a);
    var wrong = record;
    wrong.generated_tokens = 3;
    try std.testing.expectError(error.GlmKldTokenCountMismatch, validatePrompt(a, io, path, wrong, 3, 8));
    try tmp.dir.writeFile(io, .{ .sub_path = "prompts/00_code-python-topological-sort/generated_tokens.txt", .data = "1,2" });
    try std.testing.expectError(error.GlmTeacherRowTokenMismatch, validatePrompt(a, io, path, record, 3, 8));
    try tmp.dir.writeFile(io, .{ .sub_path = "prompts/00_code-python-topological-sort/generated_tokens.txt", .data = "1,3" });
    try std.testing.expectError(error.InvalidGlmDraftToken, validatePrompt(a, io, path, record, 3, 8));
    try tmp.dir.writeFile(io, .{ .sub_path = "prompts/00_code-python-topological-sort/generated_tokens.txt", .data = "1,0" });
    try tmp.dir.writeFile(io, .{ .sub_path = "prompts/00_code-python-topological-sort/logits.f32", .data = std.mem.sliceAsBytes(&rows)[0..20] });
    try std.testing.expectError(error.TeacherLogitsSizeMismatch, validatePrompt(a, io, path, record, 3, 8));
}

test "GLM KLD reuses only complete matching result with all scored rows" {
    const a = std.testing.allocator;
    const good = try std.json.Stringify.valueAlloc(a, .{ .schema = schema, .complete = true, .fingerprint = "match", .all_positions = .{ .positions = @as(usize, 2) }, .prompts = .{.{ .per_position_kld = [_]f64{ 0, 0 }, .all_positions = .{ .positions = @as(usize, 2) }, .through_first_eos = .{ .positions = @as(usize, 1) } }} }, .{});
    defer a.free(good);
    try std.testing.expect(try completedMatches(a, good, "match", 1, 2));
    try std.testing.expect(!try completedMatches(a, good, "changed", 1, 2));
    try std.testing.expect(!try completedMatches(a, good, "match", 1, 3));
    try std.testing.expect(!try completedMatches(a, "{\"complete\":false}", "match", 1, 2));
    try std.testing.expect(!try completedMatches(a, "{\"complete\":true,\"fingerprint\":\"match\"}", "match", 1, 2));
}

test "GLM KLD manifest completion and safe nested paths are required" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    try std.testing.expect(safeRelativePromptDir("prompts/00_code-python-topological-sort"));
    for ([_][]const u8{ "", "/prompts/a", "prompts\\a", "prompts//a", "prompts/./a", "prompts/../a", "../prompts", "prompts/", "prompts\x00suffix" }) |path| try std.testing.expect(!safeRelativePromptDir(path));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = buf[0..try tmp.dir.realPath(io, &buf)];
    for ([_][]const u8{ "{}", "{\"complete\":false}", "{\"complete\":1}" }) |raw| {
        try tmp.dir.writeFile(io, .{ .sub_path = "baseline.json", .data = raw });
        try std.testing.expectError(error.GlmTeacherIncomplete, validateManifestComplete(a, io, path));
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "baseline.json", .data = "{\"complete\":true}" });
    try validateManifestComplete(a, io, path);
}

test "GLM KLD accepts only explicit user-truncated two-code study" {
    const a = std.testing.allocator;
    const raw = "{\"requested_prompt_count\":4,\"completed_prompt_count\":2,\"actual_positions\":1024,\"truncated\":true,\"stopped_by_user\":true,\"stop_reason\":\"user_requested_after_two_completed_prompts\"}";
    const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer parsed.deinit();
    const reduced = try studyContract(a, parsed.value, 2);
    defer reduced.deinit(a);
    try std.testing.expect(reduced.truncated and reduced.stopped_by_user);
    try std.testing.expectEqual(@as(usize, 1024), reduced.actual_positions);
    try std.testing.expectEqualStrings("user_requested_after_two_completed_prompts", reduced.stop_reason.?);
    const empty = try std.json.parseFromSlice(std.json.Value, a, "{}", .{});
    defer empty.deinit();
    const full = try studyContract(a, empty.value, 4);
    defer full.deinit(a);
    try std.testing.expect(!full.truncated);
    try std.testing.expectEqual(@as(usize, 2048), full.actual_positions);
    try std.testing.expectError(error.BadBaselineJson, studyContract(a, empty.value, 2));
    try std.testing.expectError(error.BadBaselineJson, studyContract(a, parsed.value, 4));
    var wrong = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer wrong.deinit();
    wrong.value.object.getPtr("actual_positions").?.* = .{ .integer = 1023 };
    try std.testing.expectError(error.BadBaselineJson, studyContract(a, wrong.value, 2));
    wrong.value.object.getPtr("actual_positions").?.* = .{ .integer = 1024 };
    wrong.value.object.getPtr("stopped_by_user").?.* = .{ .bool = false };
    try std.testing.expectError(error.BadBaselineJson, studyContract(a, wrong.value, 2));
}

test "GLM KLD single completed prompt requires exact human stop contract" {
    const a = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, a, "{\"requested_prompt_count\":4,\"completed_prompt_count\":1,\"actual_positions\":512,\"truncated\":true,\"stopped_by_user\":true,\"stop_reason\":\"user_requested_one_completed_prompt\"}", .{});
    defer parsed.deinit();
    const study = try studyContract(a, parsed.value, 1);
    defer study.deinit(a);
    try std.testing.expectEqual(@as(usize, 512), study.actual_positions);
    try std.testing.expect(study.truncated and study.stopped_by_user);
    parsed.value.object.getPtr("actual_positions").?.* = .{ .integer = 1024 };
    try std.testing.expectError(error.BadBaselineJson, studyContract(a, parsed.value, 1));
    parsed.value.object.getPtr("actual_positions").?.* = .{ .integer = 512 };
    parsed.value.object.getPtr("stop_reason").?.* = .{ .string = "other" };
    try std.testing.expectError(error.BadBaselineJson, studyContract(a, parsed.value, 1));
}
