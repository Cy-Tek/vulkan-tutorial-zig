const builtin = @import("builtin");
const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const glfw = @import("glfw");
const Hints = glfw.Window.Hints;

const vk = @import("vk.zig");
const vk_ctx = @import("vk_context.zig");
const BaseDispatch = vk_ctx.BaseDispatch;
const InstanceDispatch = vk_ctx.InstanceDispatch;

const width = 800;
const height = 600;

pub const App = struct {
    const Self = @This();

    window: glfw.Window,
    alloc: Allocator,

    instance: vk.Instance,
    extensions: ArrayList([*:0]const u8),

    vkb: BaseDispatch,
    vki: InstanceDispatch,

    pub fn init(alloc: Allocator) !App {
        const window = try initWindow();
        const base_dispatch = try BaseDispatch.load(@as(
            vk.PfnGetInstanceProcAddr,
            @ptrCast(&glfw.getInstanceProcAddress),
        ));

        var app = App{
            .alloc = alloc,
            .window = window,
            .vkb = base_dispatch,
            .vki = undefined,
            .extensions = undefined,
            .instance = undefined,
        };

        try app.initVulkan();

        return app;
    }

    pub fn deinit(self: *Self) void {
        self.extensions.deinit();
        self.window.destroy();
        glfw.terminate();
    }

    pub fn run(self: *Self) !void {
        while (!self.window.shouldClose()) {
            glfw.pollEvents();
        }
    }

    fn initWindow() !glfw.Window {
        _ = glfw.init(.{});
        return glfw.Window.create(width, height, "Hello Vulkan from Zig!", null, null, .{
            .client_api = Hints.ClientAPI.no_api,
            .resizable = false,
        }).?;
    }

    fn initVulkan(self: *Self) !void {
        try self.getRequiredExtensions();
        try self.createInstance();
    }

    fn getRequiredExtensions(self: *Self) !void {
        const glfw_extensions = glfw.getRequiredInstanceExtensions() orelse return blk: {
            const err = glfw.mustGetError();
            std.log.err("Failed to get required vulkan instance extensions: error={s}", .{err.description});
            break :blk error.code;
        };

        if (builtin.mode == std.builtin.OptimizeMode.Debug) {
            var property_count: u32 = 0;
            _ = try self.vkb.enumerateInstanceExtensionProperties(null, &property_count, null);

            var extension_properties = ArrayList(vk.ExtensionProperties).init(self.alloc);
            try extension_properties.resize(property_count);
            defer extension_properties.deinit();

            _ = try self.vkb.enumerateInstanceExtensionProperties(null, &property_count, extension_properties.items.ptr);

            for (extension_properties.items) |ext| {
                std.debug.print("\nFound Extension: {s}", .{ext.extension_name});
            }
            std.debug.print("\n\n", .{});
        }

        var all_extensions = std.ArrayList([*:0]const u8).init(self.alloc);
        try all_extensions.appendSlice(glfw_extensions);

        if (builtin.os.tag == .macos) {
            try all_extensions.append(vk.extension_info.khr_portability_enumeration.name);
        }

        self.extensions = all_extensions;
    }

    fn createInstance(self: *Self) !void {
        const app_info = vk.ApplicationInfo{
            .p_application_name = "Hello Triangle",
            .application_version = vk.makeApiVersion(0, 1, 3, 0),
            .p_engine_name = "No Engine",
            .engine_version = vk.makeApiVersion(0, 1, 3, 0),
            .api_version = vk.makeApiVersion(0, 1, 3, 0),
        };

        if (builtin.mode == std.builtin.OptimizeMode.Debug) {
            for (self.extensions.items) |ext| {
                std.debug.print("Extension Required: {s}\n", .{ext});
            }
        }

        const create_info = vk.InstanceCreateInfo{
            .p_application_info = &app_info,
            .enabled_extension_count = @intCast(self.extensions.items.len),
            .pp_enabled_extension_names = self.extensions.items.ptr,
            .enabled_layer_count = 0,
            .flags = .{ .enumerate_portability_bit_khr = true },
        };

        self.instance = try self.vkb.createInstance(&create_info, null);
        self.vki = try InstanceDispatch.load(self.instance, self.vkb.dispatch.vkGetInstanceProcAddr);
    }
};
