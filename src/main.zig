const std = @import("std");
const glfw = @import("glfw");
const App = @import("app.zig").App;

fn errorCallback(error_code: glfw.ErrorCode, description: [:0]const u8) void {
    std.log.err("glfw: {}: {s}\n", .{ error_code, description });
}

pub fn main() !void {
    glfw.setErrorCallback(errorCallback);

    var app = try App.init(std.heap.page_allocator);
    defer app.deinit();

    try app.run();
}
